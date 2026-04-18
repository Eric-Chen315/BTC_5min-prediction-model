"""
train_5min_tcn.py  ——  TCN-LSTM 训练：BTC Polymarket 5分钟方向概率预测
=======================================================================

任务定义：
  给定当前 tick 的（微结构特征 + 局面状态），预测本局最终
  "close > open"（UP方向赢）的概率。

  关键设计：
    1. 状态特征（time_remaining, current_gap, gap_normalized 等）直接告诉模型
       "现在领先多少、还剩多少时间"，模型能学到：
         - gap大 + time少 → 接近1（不可能翻转）
         - gap≈0 + time多 → 接近0.5（50/50）
         - 微结构信号强 + gap同向 → 概率增强

    2. 窗口设计：
       - WINDOW = 300 tick（30秒）：捕捉当前30秒微结构趋势
       - 每秒采样1次（stride=10），共约 270局/天 × 3000tick/局 = 81000样本/天

    3. 模型架构：复用 TCN-LSTM，单头输出 prob_up
       - 输入归一化时对 current_gap 和 time_remaining 用特殊缩放（不用全局mean/std）

    4. 损失函数：BCE + Pairwise Ranking（AUC导向）
       - 不用 "3s/30s" 多horizon，只有一个目标：5min结果

用法：
  python train_5min_tcn.py
  python train_5min_tcn.py --train-start 2026-03-01 --train-end 2026-03-24 \\
                            --test-start 2026-03-25 --test-end 2026-03-30
  python train_5min_tcn.py --epochs 20 --batch 512 --stride 10
"""

from __future__ import annotations

import argparse
import gc
import os
import time
import warnings
from datetime import date as _date, timedelta as _td
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from sklearn.metrics import roc_auc_score, brier_score_loss
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 特征列（与 build_5min_dataset.py 完全一致）
# ─────────────────────────────────────────────────────────────────────────────
BID_PX_COLS = [f"bid_px{i}" for i in range(5)]
BID_SZ_COLS = [f"bid_sz{i}" for i in range(5)]
ASK_PX_COLS = [f"ask_px{i}" for i in range(5)]
ASK_SZ_COLS = [f"ask_sz{i}" for i in range(5)]
L2_COLS = BID_PX_COLS + BID_SZ_COLS + ASK_PX_COLS + ASK_SZ_COLS + ["ofi", "log_total_depth"]

TRADE_COLS = ["trade_cnt", "buy_vol", "sell_vol", "taker_imb", "vwap_ret_bps"]

DERIVED_COLS = [
    "ret_1s", "ret_5s", "ret_10s", "ret_30s", "ret_60s",
    "vol_10s", "vol_60s", "rvol_ratio",
    "ofi_r10", "ofi_r30", "ofi_r100", "ofi_norm", "ofi_l1234",
    "book_imb", "wmp_bps", "microprice_bps",
    "bid_slope", "ask_slope", "bid_ask_sz_diff",
    "queue_imb_r10", "queue_imb_r30",
    "taker_imb_r10", "taker_imb_r30",
    "kyle_lambda", "trade_intensity", "vol_momentum",
    "roll_spread", "spread_ma", "bb_pos",
    "depth_imb_w", "aggr_buy", "aggr_sell",
    "wmp_change_1s", "depth_pressure",
    "cancel_proxy_bid", "cancel_proxy_ask",
    "spread_diff",
    "vol_ratio_3_30", "vol_ratio_10_30", "vol_term_slope",
    "er_30", "er_100",
    "acf1_sf_1s", "run_ratio_sf_1s", "flip_rate_sf_1s",
    "sec_in_slot_sin", "sec_in_slot_cos",
    "is_slot_first_10s", "is_slot_last_30s",
]

MICRO_FEATURES = L2_COLS + TRADE_COLS + DERIVED_COLS   # 76个微结构特征

# 9个局面状态特征（最重要的创新）
STATE_FEATURES = [
    "time_remaining_s",   # 剩余秒数
    "time_frac",          # 已消耗时间比
    "time_sin",           # sin(π * time_frac)
    "time_cos",           # cos(π * time_frac)
    "current_gap",        # mid - open_price
    "gap_abs",            # abs(current_gap)
    "gap_sign",           # sign(current_gap)
    "gap_per_second",     # gap / (time_remaining + 1)
    "gap_normalized",     # gap / (vol * sqrt(time_remaining+1))
]

ALL_FEATURES  = MICRO_FEATURES + STATE_FEATURES
N_FEATURES    = len(ALL_FEATURES)   # 85

# 状态特征在 ALL_FEATURES 中的索引（用于特殊归一化处理）
STATE_IDX = [ALL_FEATURES.index(c) for c in STATE_FEATURES]

print(f"[config] 总特征数: {N_FEATURES}（微结构{len(MICRO_FEATURES)} + 状态{len(STATE_FEATURES)}）")

# ─────────────────────────────────────────────────────────────────────────────
# 超参数
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR  = Path(__file__).parent.parent / "DATE/5min_dataset"
MODEL_DIR    = Path(__file__).parent / "models/5min_tcn"
TRAIN_DATES_DEFAULT = ("2026-03-01", "2026-03-24")
TEST_DATES_DEFAULT  = ("2026-03-25", "2026-03-30")

WINDOW   = 300    # 30秒窗口（100ms × 300 tick）
STRIDE   = 10     # 每10tick（1s）采一个样本
BATCH    = 512
EPOCHS   = 20
LR       = 2e-4
DROPOUT  = 0.35
RANK_WEIGHT = 0.3   # Pairwise Ranking Loss 权重（AUC导向）

# TCN/LSTM 参数
TCN_CHANNELS = 128
TCN_LAYERS   = 6
KERNEL_SIZE  = 3
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 1
FC_HIDDEN    = 128

# SWA
SWA_START_EPOCH = 15   # 第15epoch后开始SWA
SWA_LR          = 1e-4


DIR_LAMBDA_3S  = 0.3   # 3s  方向任务损失权重
DIR_LAMBDA_30S = 0.4   # 30s 方向任务损失权重
DIR_GAP_SIGMA  = 20.0  # gap 权重 σ（USDT）

# ─────────────────────────────────────────────────────────────────────────────
# 数据集
# ─────────────────────────────────────────────────────────────────────────────
class FiveMinDataset(Dataset):
    """
    滑动窗口数据集。每个窗口返回：
      x            [T, 85]   特征矩阵
      label_5min   scalar    本局最终结果（0/1）
      label_dir    scalar    未来3s方向软标签（0/0.5/1）
      dir_weight   scalar    gap自适应权重（gap越大越小）
      time_weight  scalar    时序衰减权重（越新越重）
    """

    def __init__(self, df: pd.DataFrame, window: int = WINDOW, stride: int = STRIDE,
                 time_weight_tau_days: float = 5.0):
        self.window = window
        self.stride = stride

        df = df.reset_index(drop=True)

        # 特征矩阵
        self.feat_arr = df[ALL_FEATURES].values.astype(np.float32)

        # 结算标签
        lbl = df["label_5min_up"].values.astype(np.float32)
        self.label = np.clip(lbl, 0.0, 1.0)

        # 方向软标签（如果数据集已有则直接用，否则全部填 0.5）
        if "label_dir_3s" in df.columns:
            self.label_dir_3s = df["label_dir_3s"].values.astype(np.float32)
        else:
            self.label_dir_3s = np.full(len(df), 0.5, dtype=np.float32)

        if "label_dir_30s" in df.columns:
            self.label_dir_30s = df["label_dir_30s"].values.astype(np.float32)
        else:
            self.label_dir_30s = np.full(len(df), 0.5, dtype=np.float32)

        # 时间权重（越新越重要，tau=5天）
        N = len(df)
        tau_ticks = time_weight_tau_days * 864_000.0
        t_arr = np.arange(N, dtype=np.float32)
        w_raw = np.exp(-(N - 1 - t_arr) / tau_ticks)
        self.time_weight = np.clip(w_raw, 0.2, 1.0).astype(np.float32)

        # 按小时分段，防止跨小时污染
        if "date" in df.columns and "hour" in df.columns:
            group_key = df["date"].astype(str) + "_" + df["hour"].astype(str)
        elif "event_time" in df.columns:
            group_key = (df["event_time"].astype(np.int64) // 3_600_000).astype(str)
        else:
            group_key = pd.Series(["all"] * len(df), index=df.index)

        # 生成窗口索引
        self.indices: List[Tuple[int, int]] = []
        for _, g_idx in df.groupby(group_key, sort=False).groups.items():
            pos = np.asarray(g_idx)
            n = len(pos)
            if n < window:
                continue
            for i in range(window - 1, n, stride):
                p_start = int(pos[i - window + 1])
                p_end   = int(pos[i])
                self.indices.append((p_start, p_end))

        # 统计
        sample_idx = self.indices[::max(1, len(self.indices) // 10000)]
        up_rate = float(np.mean([self.label[e] for _, e in sample_idx]))
        has_dir = "label_dir_3s" in df.columns
        print(f"  → {len(self.indices):,} 个窗口样本  UP率≈{up_rate:.2%}  {'含3s/30s方向标签' if has_dir else '无方向标签(用0.5)'}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        p_start, p_end = self.indices[idx]
        x  = self.feat_arr[p_start : p_end + 1]    # [T, F]
        return (
            torch.from_numpy(x.copy()),
            torch.tensor(self.label[p_end],         dtype=torch.float32),  # label_5min
            torch.tensor(self.label_dir_3s[p_end],  dtype=torch.float32),  # label_3s
            torch.tensor(self.label_dir_30s[p_end], dtype=torch.float32),  # label_30s
            torch.tensor(self.time_weight[p_end],   dtype=torch.float32),  # 时序权重
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise Ranking Loss
# ─────────────────────────────────────────────────────────────────────────────
def pairwise_ranking_loss(scores: torch.Tensor, labels: torch.Tensor,
                          n_pairs: int = 1024) -> torch.Tensor:
    """直接优化 AUC：-log(σ(score_pos - score_neg))"""
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.0, device=scores.device)
    pos_s = scores[pos_mask]
    neg_s = scores[neg_mask]
    n = min(n_pairs, n_pos * n_neg)
    pi = torch.randint(0, n_pos, (n,), device=scores.device)
    ni = torch.randint(0, n_neg, (n,), device=scores.device)
    diff = pos_s[pi] - neg_s[ni]
    return F_nn.softplus(-diff).mean()


# ─────────────────────────────────────────────────────────────────────────────
# TCN 模块（与 train_tcn_lstm.py 相同）
# ─────────────────────────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(F_nn.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=DROPOUT):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch,  out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.skip(x)
        out = self.drop(self.relu(self.norm1(self.conv1(x))))
        out = self.drop(self.relu(self.norm2(self.conv2(out))))
        return self.relu(out + res)


# ─────────────────────────────────────────────────────────────────────────────
# 核心模型：3-Horizon 融合 TCN-LSTM
# ─────────────────────────────────────────────────────────────────────────────
class FiveMinTCNLSTM(nn.Module):
    """
    3-Horizon 多任务融合架构

    输入:  x [B, T=300, F=85]
    输出:  prob_settlement [B]  ← 5min最终结局概率（主任务）
           prob_dir_30s    [B]  ← 未来30s方向概率（辅任务，仅训练时用）
           prob_dir_3s     [B]  ← 未来3s方向概率（辅任务，仅训练时用）

    架构：
      micro_branch (TCN-LSTM on 76维)
          ├─ head_3s  → prob_dir_3s   （纯微结构，强迫 micro 学短期方向）
          └─ head_30s → prob_dir_30s  （纯微结构，强迫 micro 学中期方向）

      state_branch (FC on 9维)
          └─ state_logit               （gap/time主导的结局基准）

      dynamic_gate (由 gap_abs + time_remaining 决定权重)
          输入: [micro_feat, state_logit, gap_abs, time_frac]
          输出: prob_settlement         （动态融合，不再是全局固定 fw）

    训练损失:
      L = L_5min
        + λ_30s × mean(dir_weight × L_30s)   ← 梯度只流 micro_branch
        + λ_3s  × mean(dir_weight × L_3s)    ← 梯度只流 micro_branch

    推理时:
      prob_final = prob_settlement  （gate 已内化了 gap/time 权重）
      同时输出 prob_dir_30s 和 prob_dir_3s 供前端显示
    """

    def __init__(
        self,
        n_features:   int   = N_FEATURES,
        n_micro:      int   = len(MICRO_FEATURES),
        n_state:      int   = len(STATE_FEATURES),
        tcn_channels: int   = TCN_CHANNELS,
        tcn_layers:   int   = TCN_LAYERS,
        kernel_size:  int   = KERNEL_SIZE,
        lstm_hidden:  int   = LSTM_HIDDEN,
        fc_hidden:    int   = FC_HIDDEN,
        dropout:      float = DROPOUT,
    ):
        super().__init__()
        self.n_micro = n_micro
        self.n_state = n_state

        # ── 微结构分支（TCN-LSTM）────────────────────────────────────────
        self.micro_norm = nn.InstanceNorm1d(n_micro, affine=True)
        self.input_proj = nn.Conv1d(n_micro, tcn_channels, 1)

        tcn_blocks = []
        for i in range(tcn_layers):
            tcn_blocks.append(TCNBlock(tcn_channels, tcn_channels, kernel_size, 2**i, dropout))
        self.tcn = nn.Sequential(*tcn_blocks)

        self.lstm = nn.LSTM(
            input_size=tcn_channels, hidden_size=lstm_hidden,
            num_layers=LSTM_LAYERS, batch_first=True,
            dropout=dropout if LSTM_LAYERS > 1 else 0.0,
        )

        self.micro_skip = nn.Linear(n_micro, fc_hidden)
        micro_fused_dim = lstm_hidden + fc_hidden
        self.micro_fc = nn.Sequential(
            nn.Linear(micro_fused_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── 微结构辅助 head：3s 和 30s 方向 ─────────────────────────────
        # 这两个 head 通过辅助损失强迫 micro_branch 学到真实微结构规律
        self.head_3s  = nn.Linear(fc_hidden, 1)
        self.head_30s = nn.Linear(fc_hidden, 1)

        # ── 状态直通路径（State Branch）─────────────────────────────────
        self.state_norm = nn.LayerNorm(n_state)
        self.state_fc = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        # ── 动态融合门控（Dynamic Gate）──────────────────────────────────
        # 输入: micro_feat(fc_hidden) + state_logit(1) + gap_abs(1) + time_frac(1)
        # 输出: settlement logit
        # 门控让模型自己学：gap大+time少 → 信 state；gap小+time多 → 信 micro
        gate_in_dim = fc_hidden + 1 + 2   # micro_feat + state_logit + [gap_abs, time_frac]
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    def _micro_forward(self, x: torch.Tensor):
        """提取 micro_branch 特征（供 forward 和辅助损失共用）"""
        x_micro = x[:, :, :self.n_micro]           # [B, T, 76]
        xt = x_micro.permute(0, 2, 1)              # [B, 76, T]
        xt = self.micro_norm(xt)
        xt = self.input_proj(xt)                   # [B, C, T]
        xt = self.tcn(xt)                          # [B, C, T]
        xl = xt.permute(0, 2, 1)                   # [B, T, C]
        _, (h_n, _) = self.lstm(xl)
        h = h_n[-1]                                # [B, H]
        skip = torch.relu(self.micro_skip(x_micro[:, -1, :]))   # [B, fc_hidden]
        micro_feat = self.micro_fc(torch.cat([h, skip], dim=1)) # [B, fc_hidden]
        return micro_feat, x_micro

    def forward(self, x: torch.Tensor):
        """
        返回 (prob_settlement, prob_dir_30s, prob_dir_3s)
        训练时三个都用；推理时主要用 prob_settlement
        """
        # ── 微结构分支 ────────────────────────────────────────────────
        micro_feat, x_micro = self._micro_forward(x)

        # 辅助输出：3s 和 30s 方向概率
        prob_dir_3s  = torch.sigmoid(self.head_3s(micro_feat).squeeze(1))   # [B]
        prob_dir_30s = torch.sigmoid(self.head_30s(micro_feat).squeeze(1))  # [B]

        # ── 状态直通路径 ──────────────────────────────────────────────
        x_state = x[:, -1, self.n_micro:]          # [B, 9]，只取最后一个tick
        xs = self.state_norm(x_state)
        state_logit = self.state_fc(xs)             # [B, 1]

        # ── 动态门控融合 ──────────────────────────────────────────────
        # 从 STATE_FEATURES 中取 gap_abs(idx=5) 和 time_frac(idx=1)
        # 注意这里用的是已归一化的值（gap_abs/200, time_frac本身已[0,1]）
        gap_norm  = x_state[:, 5:6]   # gap_abs（已/200归一化）
        time_frac = x_state[:, 1:2]   # time_frac（已[0,1]）

        gate_in = torch.cat([micro_feat, state_logit, gap_norm, time_frac], dim=1)
        settlement_logit = self.gate(gate_in).squeeze(1)   # [B]

        prob_settlement = torch.sigmoid(settlement_logit)  # [B]

        return prob_settlement, prob_dir_30s, prob_dir_3s


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载工具
# ─────────────────────────────────────────────────────────────────────────────
def date_range(start_str: str, end_str: str) -> List[str]:
    s = _date.fromisoformat(start_str)
    e = _date.fromisoformat(end_str)
    return [(s + _td(days=i)).isoformat() for i in range((e - s).days + 1)]


def load_dates(dataset_dir: Path, dates: List[str], tag: str) -> pd.DataFrame:
    parts = []
    for d in dates:
        fp = dataset_dir / f"5min_{d}.parquet"
        if fp.exists():
            parts.append(pd.read_parquet(fp))
        else:
            print(f"  [{tag}] 跳过 {d}（文件不存在）")
    if not parts:
        raise FileNotFoundError(f"[{tag}] 未找到数据，请先运行 build_5min_dataset.py")
    df = pd.concat(parts, ignore_index=True)

    # 验证特征列
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"[{tag}] 缺少特征列: {missing}")

    # dropna
    drop_cols = ALL_FEATURES + ["label_5min_up"]
    n_before = len(df)
    df = df.dropna(subset=drop_cols)
    if len(df) < n_before:
        print(f"  [{tag}] dropna 去除 {n_before-len(df):,} 行")

    up_rate = df["label_5min_up"].mean()
    print(f"  [{tag}] {len(dates)}天  {len(df):,}行  UP率={up_rate:.2%}")
    return df


def compute_norm_stats(df: pd.DataFrame):
    """对微结构特征计算 robust 归一化统计（p1/p99 clip）
    状态特征不做全局归一化（已经是有物理意义的量纲）
    """
    arr = df[MICRO_FEATURES].values.astype(np.float32)
    p1  = np.percentile(arr, 1,  axis=0)
    p99 = np.percentile(arr, 99, axis=0)
    arr = np.clip(arr, p1, p99)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32), \
           p1.astype(np.float32), p99.astype(np.float32)


def apply_norm(df: pd.DataFrame, mean, std, p1, p99) -> pd.DataFrame:
    """应用微结构特征归一化，状态特征做简单缩放"""
    df = df.copy()
    arr = df[MICRO_FEATURES].values.astype(np.float32)
    arr = np.clip(arr, p1, p99)
    arr = (arr - mean) / std
    df[MICRO_FEATURES] = arr

    # 状态特征简单缩放到合理量级
    df["time_remaining_s"] = df["time_remaining_s"] / 300.0   # → [0,1]
    df["time_frac"]        = df["time_frac"]                   # 已[0,1]
    df["current_gap"]      = df["current_gap"] / 200.0        # BTC±200 USDT 为合理范围
    df["gap_abs"]          = df["gap_abs"] / 200.0
    df["gap_per_second"]   = df["gap_per_second"] / 10.0      # ±10 USDT/s
    # gap_normalized 已经是标准化值，clip到[-5,5]已在构建时处理
    # gap_sign, time_sin, time_cos 已在[-1,1]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 训练一个 Epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, rank_weight=RANK_WEIGHT):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        x, lbl_5min, lbl_3s, lbl_30s, tw = [b.to(device) for b in batch]

        optimizer.zero_grad()
        prob_settlement, prob_dir_30s, prob_dir_3s = model(x)

        # ── 主任务：5min结算结局 ──────────────────────────────────────
        bce = F_nn.binary_cross_entropy(prob_settlement, lbl_5min,
                                        weight=tw, reduction="mean")
        rank = pairwise_ranking_loss(prob_settlement, lbl_5min)
        L_main = (1 - rank_weight) * bce + rank_weight * rank

        # ── 辅任务：方向软标签（梯度只流 micro_branch）────────────────
        # 只在非平盘样本（label≠0.5）上计算方向损失
        mask_3s  = (lbl_3s  != 0.5)
        mask_30s = (lbl_30s != 0.5)

        if mask_3s.sum() > 0:
            L_3s = F_nn.binary_cross_entropy(
                prob_dir_3s[mask_3s], lbl_3s[mask_3s], reduction="mean")
        else:
            L_3s = torch.tensor(0.0, device=device)

        if mask_30s.sum() > 0:
            L_30s = F_nn.binary_cross_entropy(
                prob_dir_30s[mask_30s], lbl_30s[mask_30s], reduction="mean")
        else:
            L_30s = torch.tensor(0.0, device=device)

        loss = L_main + DIR_LAMBDA_3S * L_3s + DIR_LAMBDA_30S * L_30s
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 评估：AUC + Brier Score + 校准检验
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, tag="eval"):
    model.eval()
    all_probs  = []
    all_labels = []
    all_3s_probs  = []
    all_3s_labels = []
    all_30s_probs  = []
    all_30s_labels = []

    for batch in loader:
        x, lbl_5min, lbl_3s, lbl_30s, _ = [b.to(device) for b in batch]
        prob_settlement, prob_dir_30s, prob_dir_3s = model(x)
        all_probs.append(prob_settlement.cpu().numpy())
        all_labels.append(lbl_5min.cpu().numpy())
        all_3s_probs.append(prob_dir_3s.cpu().numpy())
        all_3s_labels.append(lbl_3s.cpu().numpy())
        all_30s_probs.append(prob_dir_30s.cpu().numpy())
        all_30s_labels.append(lbl_30s.cpu().numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    probs_3s  = np.concatenate(all_3s_probs)
    labels_3s = np.concatenate(all_3s_labels)
    probs_30s  = np.concatenate(all_30s_probs)
    labels_30s = np.concatenate(all_30s_labels)

    auc   = roc_auc_score(labels, probs)
    brier = brier_score_loss(labels, probs)

    # 方向AUC（只在非平盘样本上计算）
    mask_3s  = labels_3s  != 0.5
    mask_30s = labels_30s != 0.5
    auc_3s  = roc_auc_score(labels_3s[mask_3s].round(),   probs_3s[mask_3s])   if mask_3s.sum()  > 10 else float("nan")
    auc_30s = roc_auc_score(labels_30s[mask_30s].round(), probs_30s[mask_30s]) if mask_30s.sum() > 10 else float("nan")

    print(f"  [{tag}]  AUC={auc:.4f}  Brier={brier:.4f}  "
          f"AUC_3s={auc_3s:.4f}  AUC_30s={auc_30s:.4f}")

    # 概率校准检验：按10分位查看实际UP率 vs 预测概率
    bins = np.percentile(probs, np.linspace(0, 100, 11))
    bins[0]  -= 1e-6
    bins[-1] += 1e-6
    calib_str = ""
    for i in range(10):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() > 0:
            actual = labels[mask].mean()
            pred   = probs[mask].mean()
            calib_str += f"[{pred:.2f}→{actual:.2f}] "

    print(f"  [{tag}]  校准: {calib_str}")

    # 高置信度场景统计（实战最关心的）
    for thr in [0.75, 0.85, 0.90, 0.95]:
        mask_hi = probs > thr
        mask_lo = probs < (1 - thr)
        hi_acc  = labels[mask_hi].mean() if mask_hi.sum() > 0 else float("nan")
        lo_acc  = (1 - labels[mask_lo]).mean() if mask_lo.sum() > 0 else float("nan")
        print(f"  [{tag}]  P>{thr:.0%}: {mask_hi.sum()}个 实际UP={hi_acc:.2%}  "
              f"P<{1-thr:.0%}: {mask_lo.sum()}个 实际DOWN={lo_acc:.2%}")

    return auc, brier


# ─────────────────────────────────────────────────────────────────────────────
# 主训练流程
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-start",  default=TRAIN_DATES_DEFAULT[0])
    ap.add_argument("--train-end",    default=TRAIN_DATES_DEFAULT[1])
    ap.add_argument("--test-start",   default=TEST_DATES_DEFAULT[0])
    ap.add_argument("--test-end",     default=TEST_DATES_DEFAULT[1])
    ap.add_argument("--data",         default=str(DATASET_DIR))
    ap.add_argument("--out",          default=str(MODEL_DIR))
    ap.add_argument("--epochs",       type=int,   default=EPOCHS)
    ap.add_argument("--batch",        type=int,   default=BATCH)
    ap.add_argument("--stride",       type=int,   default=STRIDE)
    ap.add_argument("--window",       type=int,   default=WINDOW)
    ap.add_argument("--lr",           type=float, default=LR)
    ap.add_argument("--rank-weight",  type=float, default=RANK_WEIGHT)
    ap.add_argument("--no-swa",       action="store_true")
    args = ap.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data)

    print(f"\n{'='*60}")
    print(f"BTC 5分钟方向概率 TCN-LSTM 训练")
    print(f"  设备: {device}  特征数: {N_FEATURES}")
    print(f"  训练: {args.train_start}~{args.train_end}")
    print(f"  测试: {args.test_start}~{args.test_end}")
    print(f"  epochs={args.epochs}  batch={args.batch}  stride={args.stride}")
    print(f"{'='*60}\n")

    # ── 加载数据 ──────────────────────────────────────────────────────
    print("[1/5] 加载数据...")
    train_dates = date_range(args.train_start, args.train_end)
    test_dates  = date_range(args.test_start,  args.test_end)
    train_df = load_dates(data_dir, train_dates, "train")
    test_df  = load_dates(data_dir, test_dates,  "test")

    # ── 归一化 ────────────────────────────────────────────────────────
    print("\n[2/5] 计算归一化统计（训练集）...")
    mean, std, p1, p99 = compute_norm_stats(train_df)
    np.save(out_dir / "norm_mean.npy", mean)
    np.save(out_dir / "norm_std.npy",  std)
    np.save(out_dir / "norm_p1.npy",   p1)
    np.save(out_dir / "norm_p99.npy",  p99)
    print(f"  微结构均值范围: [{mean.min():.3f}, {mean.max():.3f}]")

    train_df = apply_norm(train_df, mean, std, p1, p99)
    test_df  = apply_norm(test_df,  mean, std, p1, p99)

    # ── 构建 Dataset ──────────────────────────────────────────────────
    print("\n[3/5] 构建滑动窗口数据集...")
    train_ds = FiveMinDataset(train_df, window=args.window, stride=args.stride)
    test_ds  = FiveMinDataset(test_df,  window=args.window, stride=args.stride * 3)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch * 2, shuffle=False,
                              num_workers=2, pin_memory=True)

    del train_df, test_df
    gc.collect()

    # ── 构建模型 ──────────────────────────────────────────────────────
    print("\n[4/5] 初始化模型...")
    model = FiveMinTCNLSTM().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {n_params:,}")
    print(f"  架构: micro_branch(TCN-LSTM) + state_branch(FC) + GateNetwork(动态融合)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 3,
        steps_per_epoch=len(train_loader),
        epochs=min(args.epochs, SWA_START_EPOCH),   # OneCycle 只跑到SWA开始前
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    # SWA
    use_swa = (not args.no_swa) and (args.epochs >= SWA_START_EPOCH)
    if use_swa:
        swa_model = AveragedModel(model)
        swa_sched  = SWALR(optimizer, swa_lr=SWA_LR, anneal_epochs=3)
        print(f"  SWA 启用（epoch {SWA_START_EPOCH}+）")

    best_auc = 0.0

    # ── 训练循环 ──────────────────────────────────────────────────────
    print("\n[5/5] 开始训练...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        in_swa = use_swa and (epoch >= SWA_START_EPOCH)

        train_loss = train_epoch(model, train_loader, optimizer, device, args.rank_weight)

        if not in_swa:
            scheduler.step()
        else:
            swa_model.update_parameters(model)
            swa_sched.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}/{args.epochs}  loss={train_loss:.4f}  "
              f"lr={current_lr:.2e}  [{elapsed:.0f}s]")

        # 每2epoch评估一次（+ 最后一个epoch）
        if epoch % 2 == 0 or epoch == args.epochs:
            eval_model = swa_model if in_swa else model
            if in_swa:
                # SWA 需要更新 BN 统计（InstanceNorm不需要，但保险起见）
                torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            auc, brier = evaluate(eval_model, test_loader, device, tag=f"test-ep{epoch}")

            if auc > best_auc:
                best_auc = auc
                # 保存最佳模型
                save_obj = {
                    "epoch":         epoch,
                    "auc":           auc,
                    "brier":         brier,
                    "state_dict":    (swa_model if in_swa else model).state_dict(),
                    "n_features":    N_FEATURES,
                    "n_micro":       len(MICRO_FEATURES),
                    "n_state":       len(STATE_FEATURES),
                    "all_features":  ALL_FEATURES,
                    "norm_mean":     mean,
                    "norm_std":      std,
                    "norm_p1":       p1,
                    "norm_p99":      p99,
                    "window":        args.window,
                    "train_start":   args.train_start,
                    "train_end":     args.train_end,
                }
                torch.save(save_obj, out_dir / "best_5min_model.pt")
                print(f"  ★ 最佳模型已保存  AUC={auc:.4f}  Brier={brier:.4f}")

        print()

    print(f"\n{'='*60}")
    print(f"训练完成！最佳测试 AUC = {best_auc:.4f}")
    print(f"模型保存于: {out_dir / 'best_5min_model.pt'}")
    print(f"{'='*60}")

    # ── 最终评估：按时间段分析 ────────────────────────────────────────
    print("\n[bonus] 各epoch测试集中高置信度准确率（实战参考）：")
    print("  P>90%时的实际UP率越高 → 出手信号越可靠")
    print("  Brier Score越低 → 概率校准越准确")


if __name__ == "__main__":
    main()
