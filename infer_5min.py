"""
infer_5min.py  ——  BTC 5分钟方向实时推理服务（真实币安 WSS 版）
=============================================================================

功能：
  1. Binance WebSocket 接收 BTC depthUpdate@100ms + trade 流
  2. 标准 L2 订单簿增量重建（与 build_l2_dataset.py numba 核心等价）
  3. 每 100ms 聚合一个 tick，计算全部 85 个特征（完全对齐训练数据）
  4. 维护 300-tick 滑动窗口，每秒推理一次，广播 JSON 给前端

关键设计（与训练完全对齐）：
  - open_price = 每个 5min-slot 内第一个 tick 的 mid
                 对应 build_5min_dataset.py:  groupby('slot_ms').first()
  - time_remaining = (slot_ms + 300_000 - bucket_ms) / 1000.0
                     对应 build_5min_dataset.py 的 slot_end_ms - event_time
  - L2 快照同步：REST 快照 → 等 depthUpdate 追平 lastUpdateId → 再开始 tick
  - OFI：prev_bid5/prev_ask5 在快照后从真实盘口初始化（不从 0 开始）

输出 JSON（每秒广播）：
  {
    "ts": 1712000000000,
    "slot_ms": ...,
    "open_price": 82000.0,
    "mid": 82050.0,
    "current_gap": 50.0,
    "time_remaining_s": 240.0,
    "prob_up": 0.82,
    "prob_up_raw": 0.85,
    "n_ticks": 150,
    "model_ready": true,
    "history": [...]
  }

用法：
  python infer_5min.py
  python infer_5min.py --model DATE/models/5min_tcn/calibrated_5min_model.pt
  python infer_5min.py --port 8765 --no-ws
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 可选依赖
# ─────────────────────────────────────────────────────────────────────────────
try:
    import websockets
    HAS_WS = True
except ImportError:
    HAS_WS = False
    print("[warn] websockets 未安装：pip install websockets")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    print("[warn] aiohttp 未安装：pip install aiohttp")

from train_5min_tcn import (
    FiveMinTCNLSTM,
    ALL_FEATURES,
    MICRO_FEATURES,
    STATE_FEATURES,
    N_FEATURES,
    WINDOW,
)

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────
BINANCE_WS_URL  = "wss://stream.binance.com:9443/ws"
BINANCE_REST    = "https://api.binance.com/api/v3"
SYMBOL          = "BTCUSDT"
SYMBOL_LOWER    = "btcusdt"
N_LEVELS        = 5
TICK_MS         = 100           # 100ms bucket = 1 tick（与训练一致）
SLOT_MS         = 300_000       # 5 分钟 = 300_000 ms
INFER_EVERY_MS  = 100           # 每 100ms 推理一次，与 tick 对齐
MIN_TICKS       = 30            # 最少 30 tick（3s）才推理
HISTORY_LEN     = 300           # 保留 300 条历史（5分钟）

DEFAULT_MODEL   = str(Path(__file__).parent / "models/5min_tcn/best_5min_model.pt")
WS_PORT         = 19198         # WebSocket 推理数据端口
HTTP_PORT       = 19199         # dashboard.html HTTP 端口

# ─────────────────────────────────────────────────────────────────────────────
# L2 订单簿
# key = round(price * 100) → qty
# ─────────────────────────────────────────────────────────────────────────────
class OrderBook:
    """实时 L2 订单簿，与 build_l2_dataset.py _replay_l2_core 等价"""

    def __init__(self):
        self.bids: dict[int, float] = {}
        self.asks: dict[int, float] = {}
        self.last_update_id: int = 0

    def apply_snapshot(self, data: dict):
        """应用 REST /depth 快照"""
        self.bids.clear()
        self.asks.clear()
        for p, q in data.get("bids", []):
            key = round(float(p) * 100)
            qty = float(q)
            if qty > 0:
                self.bids[key] = qty
        for p, q in data.get("asks", []):
            key = round(float(p) * 100)
            qty = float(q)
            if qty > 0:
                self.asks[key] = qty
        self.last_update_id = int(data.get("lastUpdateId", 0))

    def apply_update(self, bids: list, asks: list, final_update_id: int = 0):
        """应用 depthUpdate 增量（qty=0 删除；qty>0 更新）"""
        for price_s, qty_s in bids:
            key = round(float(price_s) * 100)
            qty = float(qty_s)
            if qty == 0.0:
                self.bids.pop(key, None)
            else:
                self.bids[key] = qty
        for price_s, qty_s in asks:
            key = round(float(price_s) * 100)
            qty = float(qty_s)
            if qty == 0.0:
                self.asks.pop(key, None)
            else:
                self.asks[key] = qty
        if final_update_id:
            self.last_update_id = final_update_id

    def top(self, n: int = N_LEVELS):
        """返回前 n 档 bid/ask"""
        if not self.bids or not self.asks:
            return None, None, None, None
        bkeys = sorted(self.bids.keys(), reverse=True)[:n]
        akeys = sorted(self.asks.keys())[:n]
        bp = np.array([k / 100.0 for k in bkeys], dtype=np.float64)
        bq = np.array([self.bids[k] for k in bkeys], dtype=np.float64)
        ap = np.array([k / 100.0 for k in akeys], dtype=np.float64)
        aq = np.array([self.asks[k] for k in akeys], dtype=np.float64)
        if len(bp) < n:
            bp = np.pad(bp, (0, n - len(bp)), constant_values=np.nan)
            bq = np.pad(bq, (0, n - len(bq)), constant_values=0.0)
        if len(ap) < n:
            ap = np.pad(ap, (0, n - len(ap)), constant_values=np.nan)
            aq = np.pad(aq, (0, n - len(aq)), constant_values=0.0)
        return bp, bq, ap, aq

    @property
    def mid(self) -> float:
        if not self.bids or not self.asks:
            return float("nan")
        return (max(self.bids) / 100.0 + min(self.asks) / 100.0) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# 100ms Tick 聚合器
# ─────────────────────────────────────────────────────────────────────────────
class TickAggregator:
    """
    将 Binance depthUpdate + trade 聚合为 100ms tick。

    同步流程（Binance 官方文档）：
      1. 建立 WS 连接，开始缓冲 depthUpdate 到 _pending_updates
      2. 调用 REST /depth 取快照
      3. 丢弃 u <= lastUpdateId 的 pending updates
      4. 应用快照 + 回放剩余 pending
      5. _syncing=False → 正常产出 tick
    """

    def __init__(self):
        self.ob = OrderBook()

        # OFI：上一个 bucket 的 bid/ask 5档总量（快照后初始化，不从 0 开始）
        self._prev_bid5_sum: float = 0.0
        self._prev_ask5_sum: float = 0.0
        self._ob_initialized: bool = False

        # 当前 bucket 累计
        self._bucket_ms:      int   = 0
        self._buy_vol:        float = 0.0
        self._sell_vol:       float = 0.0
        self._trade_cnt:      int   = 0
        self._vwap_num:       float = 0.0
        self._vwap_den:       float = 0.0
        self._last_trade_ms:  int   = 0
        self.last_trade_price: float = float("nan")  # 最新成交价，用于前端实时显示

        # 完成 tick 队列
        self.completed: Deque[dict] = collections.deque(maxlen=10000)

        # 快照同步缓冲
        self._pending_updates: list[dict] = []
        self._syncing: bool = True

    def apply_snapshot(self, data: dict):
        """应用 REST 快照，回放缓冲区"""
        self.ob.apply_snapshot(data)
        last_id = self.ob.last_update_id

        # 用真实盘口初始化 OFI 基准（不从 0 开始）
        bp, bq, ap, aq = self.ob.top(N_LEVELS)
        if bp is not None:
            bq_safe = np.where(np.isnan(bq), 0.0, bq)
            aq_safe = np.where(np.isnan(aq), 0.0, aq)
            self._prev_bid5_sum = float(bq_safe.sum())
            self._prev_ask5_sum = float(aq_safe.sum())

        # 回放快照之后的 pending updates
        # 官方规则：丢弃 u <= lastUpdateId；第一条有效 update 必须满足
        #   U <= lastUpdateId + 1 <= u（即连续，无 gap）
        applied = 0
        first_valid = True
        for upd in self._pending_updates:
            u_val = upd.get("u", 0)
            U_val = upd.get("U", 0)
            if u_val <= last_id:
                continue   # 已包含在快照里
            if first_valid:
                if not (U_val <= last_id + 1 <= u_val):
                    # gap：快照和第一条 pending 之间有缺口，需重新同步
                    print(
                        f"[snapshot] ⚠️  gap 检测：U={U_val} lastUpdateId={last_id} u={u_val}，"
                        f"将在下次重连后重新同步",
                        flush=True,
                    )
                    self._pending_updates.clear()
                    # 保持 _syncing=True，触发断线重连
                    return
                first_valid = False
            self.ob.apply_update(upd.get("b", []), upd.get("a", []), u_val)
            applied += 1
        self._pending_updates.clear()

        self._syncing = False
        self._ob_initialized = True
        self._bucket_ms = 0   # 从当前时刻重新计 bucket
        print(f"[snapshot] 同步完成 lastUpdateId={last_id}  回放 {applied} 条 pending", flush=True)

    def on_depth_update(self, msg: dict):
        if self._syncing:
            self._pending_updates.append(msg)
            return
        et = msg.get("E", int(time.time() * 1000))
        u  = msg.get("u", 0)
        self.ob.apply_update(msg.get("b", []), msg.get("a", []), u)
        self._maybe_flush(et)

    def on_trade(self, msg: dict):
        if self._syncing:
            return
        is_buy_maker = msg.get("m", False)
        qty    = float(msg.get("q", 0))
        price  = float(msg.get("p", 0))
        t_time = int(msg.get("T", int(time.time() * 1000)))
        if is_buy_maker:
            self._sell_vol += qty
        else:
            self._buy_vol  += qty
        self._trade_cnt += 1
        self._vwap_num  += qty * price
        self._vwap_den  += qty
        self._last_trade_ms   = t_time
        self.last_trade_price = price   # 实时记录最新成交价
        self._maybe_flush(t_time)

    def _maybe_flush(self, event_ms: int):
        bucket = (event_ms // TICK_MS) * TICK_MS
        if self._bucket_ms == 0:
            self._bucket_ms = bucket
            return
        if bucket > self._bucket_ms:
            self._flush_tick()
            self._bucket_ms = bucket

    def _flush_tick(self):
        bp, bq, ap, aq = self.ob.top(N_LEVELS)
        if bp is None:
            return
        # 严格与训练数据对齐：mid = (best_bid + best_ask) / 2，纯盘口中间价
        ob_mid = self.ob.mid
        if math.isnan(ob_mid):
            return
        mid = ob_mid

        bq_safe = np.where(np.isnan(bq), 0.0, bq)
        aq_safe = np.where(np.isnan(aq), 0.0, aq)

        # OFI = Δ(bid5_total) - Δ(ask5_total)，与 _replay_l2_core 完全一致
        cur_bid5_sum = float(bq_safe.sum())
        cur_ask5_sum = float(aq_safe.sum())
        ofi = (cur_bid5_sum - self._prev_bid5_sum) - \
              (cur_ask5_sum - self._prev_ask5_sum)
        self._prev_bid5_sum = cur_bid5_sum
        self._prev_ask5_sum = cur_ask5_sum

        # stale 判断（成交超过 1s 视为空）
        # 与训练对齐：stale 时 vwap 置 nan，vwap_ret_bps 将输出 0.0
        is_stale = self._trade_cnt == 0 or (self._bucket_ms - self._last_trade_ms) > 1000
        if is_stale:
            buy_vol = sell_vol = 0.0
            trade_cnt = 0.0
            taker_imb = 0.0
            vwap = float("nan")   # stale → vwap_ret_bps = 0.0（与训练一致）
        else:
            vwap      = self._vwap_num / self._vwap_den if self._vwap_den > 0 else float("nan")
            buy_vol   = self._buy_vol
            sell_vol  = self._sell_vol
            trade_cnt = float(self._trade_cnt)
            total_vol = buy_vol + sell_vol
            taker_imb = (buy_vol - sell_vol) / (total_vol + 1e-9)

        self.completed.append({
            "ts":        self._bucket_ms,   # bucket 起始 ms（= 训练的 event_time）
            "mid":       mid,
            "bid_px":    bp.copy(),
            "bid_sz":    bq_safe.copy(),
            "ask_px":    ap.copy(),
            "ask_sz":    aq_safe.copy(),
            "ofi":       float(ofi),
            "buy_vol":   float(buy_vol),
            "sell_vol":  float(sell_vol),
            "trade_cnt": float(trade_cnt),
            "taker_imb": float(taker_imb),
            "vwap":      float(vwap),
        })

        # 重置 bucket 累计
        self._buy_vol   = 0.0
        self._sell_vol  = 0.0
        self._trade_cnt = 0
        self._vwap_num  = 0.0
        self._vwap_den  = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 特征引擎
# ─────────────────────────────────────────────────────────────────────────────
class FeatureEngine:
    """
    维护滑动窗口，计算 85 维特征矩阵，与训练完全对齐。

    核心对齐点：
      1. open_price 按 slot_ms 分组记录，使用每局第一个 tick 的 mid
         = build_5min_dataset.py groupby('slot_ms')['mid'].first()
      2. time_remaining = (slot_ms + 300_000 - bucket_ms) / 1000
         = build_5min_dataset.py 的 slot_end_ms - event_time
      3. 归一化逻辑：apply_norm 与 train_5min_tcn.py 完全一致
    """

    def __init__(self, max_ticks: int = 3000 + WINDOW):  # 3000 for vol_300s rolling, WINDOW for inference
        self.ticks: Deque[dict] = collections.deque(maxlen=max_ticks)
        # slot_ms → open_price（每局第一个 tick 的 mid）
        self._slot_open: dict[int, float] = {}
        self._current_slot_ms: int = 0

    def push(self, tick: dict):
        self.ticks.append(tick)
        ts   = tick["ts"]
        slot = (ts // SLOT_MS) * SLOT_MS

        # 每个 slot 只记录第一个 tick 的 mid 作为 open_price
        if slot not in self._slot_open:
            self._slot_open[slot] = tick["mid"]
            if slot != self._current_slot_ms and self._current_slot_ms != 0:
                print(f"[slot] 新局 {_fmt_slot(slot)}  open={tick['mid']:.2f}", flush=True)
        self._current_slot_ms = slot

        # 只保留最近 3 个 slot 的 open_price
        if len(self._slot_open) > 3:
            del self._slot_open[min(self._slot_open)]

    def get_open_price(self, slot_ms: int) -> float:
        return self._slot_open.get(slot_ms, float("nan"))

    def build_feature_window(
        self,
        window: int = WINDOW,
        norm_params: dict = None,
    ) -> Optional[np.ndarray]:
        """构建 [T, 85] 特征矩阵，完全对齐训练数据"""
        all_ticks = list(self.ticks)
        if len(all_ticks) < MIN_TICKS:
            return None

        # vol_300s 需要完整长期历史（最多 3000 tick），单独提取全量 mid
        long_mid = np.array([t["mid"] for t in all_ticks], dtype=np.float64)

        ticks = all_ticks[-window:]
        T = len(ticks)

        # ── 原始序列 ─────────────────────────────────────────────────────────
        mid    = np.array([t["mid"]       for t in ticks], dtype=np.float64)
        ofi    = np.array([t["ofi"]       for t in ticks], dtype=np.float32)
        buy_v  = np.array([t["buy_vol"]   for t in ticks], dtype=np.float32)
        sell_v = np.array([t["sell_vol"]  for t in ticks], dtype=np.float32)
        tc     = np.array([t["trade_cnt"] for t in ticks], dtype=np.float32)
        timb   = np.array([t["taker_imb"] for t in ticks], dtype=np.float32)
        vwap_a = np.array([t["vwap"]      for t in ticks], dtype=np.float64)
        ts_arr = np.array([t["ts"]        for t in ticks], dtype=np.int64)

        bid_px = np.array([t["bid_px"] for t in ticks], dtype=np.float64)  # [T,5]
        bid_sz = np.array([t["bid_sz"] for t in ticks], dtype=np.float64)
        ask_px = np.array([t["ask_px"] for t in ticks], dtype=np.float64)
        ask_sz = np.array([t["ask_sz"] for t in ticks], dtype=np.float64)

        # ── L2 归一化（与 replay_l2_vectorized 完全一致）──────────────────────
        mid_safe = np.where(mid > 0, mid, np.nan)
        bps = 10000.0

        bid_px_rel = (bid_px - mid_safe[:, None]) / mid_safe[:, None] * bps
        ask_px_rel = (ask_px - mid_safe[:, None]) / mid_safe[:, None] * bps

        bid_sz_clean = np.where(np.isnan(bid_sz), 0.0, bid_sz)
        ask_sz_clean = np.where(np.isnan(ask_sz), 0.0, ask_sz)
        total_depth_raw = bid_sz_clean.sum(1) + ask_sz_clean.sum(1) + 1e-9
        bid_sz_rel = bid_sz_clean / total_depth_raw[:, None]
        ask_sz_rel = ask_sz_clean / total_depth_raw[:, None]
        bid_sz_log = np.log1p(bid_sz_rel)
        ask_sz_log = np.log1p(ask_sz_rel)
        log_total_depth = np.log1p(total_depth_raw)
        spread_bps_arr  = (ask_px[:, 0] - bid_px[:, 0]) / mid_safe * bps

        for k in range(N_LEVELS):
            fill_bid = -(k + N_LEVELS + 1) * 5.0
            fill_ask = +(k + N_LEVELS + 1) * 5.0
            bid_px_rel[:, k] = np.where(np.isnan(bid_px_rel[:, k]), fill_bid, bid_px_rel[:, k])
            ask_px_rel[:, k] = np.where(np.isnan(ask_px_rel[:, k]), fill_ask, ask_px_rel[:, k])

        # vwap_ret_bps
        prev_mid  = np.roll(mid, 1); prev_mid[0] = mid[0]
        safe_prev = np.where(prev_mid > 0, prev_mid, np.nan)
        vwap_ret_bps = np.where(
            ~np.isnan(vwap_a),
            (vwap_a - safe_prev) / safe_prev * bps,
            0.0
        ).astype(np.float32)

        # ── DataFrame ───────────────────────────────────────────────────────
        import pandas as pd

        feat: dict = {}
        for k in range(N_LEVELS):
            feat[f"bid_px{k}"] = bid_px_rel[:, k].astype(np.float32)
            feat[f"bid_sz{k}"] = bid_sz_log[:, k].astype(np.float32)
            feat[f"ask_px{k}"] = ask_px_rel[:, k].astype(np.float32)
            feat[f"ask_sz{k}"] = ask_sz_log[:, k].astype(np.float32)
        feat["ofi"]             = ofi
        feat["log_total_depth"] = log_total_depth.astype(np.float32)
        feat["trade_cnt"]       = tc
        feat["buy_vol"]         = buy_v
        feat["sell_vol"]        = sell_v
        feat["taker_imb"]       = timb
        feat["vwap_ret_bps"]    = vwap_ret_bps
        feat["mid"]             = mid
        feat["_spread_bps"]     = spread_bps_arr.astype(np.float32)
        feat["event_time"]      = ts_arr

        df = pd.DataFrame(feat)

        # ── add_rolling_features（与 build_l2_dataset.py 完全一致）───────────
        log_mid = np.log(pd.Series(mid).replace(0, np.nan))
        lr = log_mid.diff()

        df["ret_1s"]  = lr.rolling(10,  min_periods=1).sum().astype(np.float32)
        df["ret_5s"]  = lr.rolling(50,  min_periods=1).sum().astype(np.float32)
        df["ret_10s"] = lr.rolling(100, min_periods=1).sum().astype(np.float32)
        df["ret_30s"] = lr.rolling(300, min_periods=1).sum().astype(np.float32)
        df["ret_60s"] = lr.rolling(600, min_periods=1).sum().astype(np.float32)

        vol_3s  = lr.rolling(30,  min_periods=3).std()
        vol_30s = lr.rolling(300, min_periods=10).std()
        vol_5s  = lr.rolling(50,  min_periods=3).std()
        df["vol_10s"] = lr.rolling(100, min_periods=5).std().astype(np.float32)
        df["vol_60s"] = lr.rolling(600, min_periods=10).std().astype(np.float32)
        df["rvol_ratio"] = (vol_5s / (df["vol_60s"].replace(0, np.nan) + 1e-10)
                            ).clip(0, 10).fillna(1).astype(np.float32)
        df["vol_ratio_3_30"]  = (vol_3s / (vol_30s + 1e-10)).clip(0, 10).fillna(1).astype(np.float32)
        df["vol_ratio_10_30"] = (df["vol_10s"] / (vol_30s + 1e-10)).clip(0, 10).fillna(1).astype(np.float32)
        df["vol_term_slope"]  = (np.log(vol_3s + 1e-10) - np.log(vol_30s + 1e-10)
                                 ).clip(-5, 5).fillna(0).astype(np.float32)

        mid_ser      = pd.Series(mid)
        mid_abs_diff = mid_ser.diff().abs()
        df["er_30"]  = (mid_ser.diff(30).abs() / (mid_abs_diff.rolling(30, min_periods=3).sum() + 1e-6)
                        ).clip(0, 1).fillna(0).astype(np.float32)
        df["er_100"] = (mid_ser.diff(100).abs() / (mid_abs_diff.rolling(100, min_periods=5).sum() + 1e-6)
                        ).clip(0, 1).fillna(0).astype(np.float32)

        ofi_s = pd.Series(ofi)
        df["ofi_r10"]  = ofi_s.rolling(10,  min_periods=1).mean().astype(np.float32)
        df["ofi_r30"]  = ofi_s.rolling(30,  min_periods=1).mean().astype(np.float32)
        df["ofi_r100"] = ofi_s.rolling(100, min_periods=1).mean().astype(np.float32)
        ofi_mean = ofi_s.rolling(100, min_periods=5).mean()
        ofi_std  = ofi_s.rolling(100, min_periods=5).std().replace(0, np.nan)
        df["ofi_norm"] = ((ofi_s - ofi_mean) / ofi_std).fillna(0).astype(np.float32)

        ofi_l1234_parts = []
        for k in range(1, N_LEVELS):
            bsz_k = np.expm1(bid_sz_log[:, k].clip(0))
            asz_k = np.expm1(ask_sz_log[:, k].clip(0))
            ofi_l1234_parts.append((pd.Series(bsz_k).diff() - pd.Series(asz_k).diff()).fillna(0))
        df["ofi_l1234"] = sum(ofi_l1234_parts).rolling(30, min_periods=1).mean().astype(np.float32)

        bid0 = np.expm1(bid_sz_log[:, 0].clip(0))
        ask0 = np.expm1(ask_sz_log[:, 0].clip(0))
        bid_total = sum(np.expm1(bid_sz_log[:, k].clip(0)) for k in range(N_LEVELS))
        ask_total = sum(np.expm1(ask_sz_log[:, k].clip(0)) for k in range(N_LEVELS))

        bid0_s = pd.Series(bid0); ask0_s = pd.Series(ask0)
        df["book_imb"] = ((bid0_s - ask0_s) / (bid0_s + ask0_s + 1e-9)).astype(np.float32)

        bid_px0_abs = mid_safe * (1 + bid_px_rel[:, 0] / bps)
        ask_px0_abs = mid_safe * (1 + ask_px_rel[:, 0] / bps)
        wmp = (ask_px0_abs * bid0 + bid_px0_abs * ask0) / (bid0 + ask0 + 1e-9)
        df["wmp_bps"] = ((wmp - mid_safe) / mid_safe * bps).clip(-50, 50).astype(np.float32)

        spread_abs = ask_px0_abs - bid_px0_abs
        micro = bid_px0_abs + spread_abs * (bid0 / (bid0 + ask0 + 1e-9))
        df["microprice_bps"] = ((micro - mid_safe) / mid_safe * bps).clip(-50, 50).astype(np.float32)

        bid_span = np.where(bid_px_rel[:, 4] > -50.0, bid_px_rel[:, 0] - bid_px_rel[:, 4], np.nan)
        ask_span = np.where(ask_px_rel[:, 4] <  50.0, ask_px_rel[:, 4] - ask_px_rel[:, 0], np.nan)
        df["bid_slope"] = pd.Series(bid_span / (log_total_depth + 1e-9)).fillna(0).astype(np.float32)
        df["ask_slope"] = pd.Series(ask_span / (log_total_depth + 1e-9)).fillna(0).astype(np.float32)

        df["bid_ask_sz_diff"] = ((bid_total - ask_total) * 2.0).clip(-1, 1).astype(np.float32)

        bi = df["book_imb"]
        df["queue_imb_r10"] = bi.rolling(10, min_periods=1).mean().astype(np.float32)
        df["queue_imb_r30"] = bi.rolling(30, min_periods=1).mean().astype(np.float32)

        ti = pd.Series(timb)
        df["taker_imb_r10"] = ti.rolling(10, min_periods=1).mean().astype(np.float32)
        df["taker_imb_r30"] = ti.rolling(30, min_periods=1).mean().astype(np.float32)

        trade_vol     = pd.Series(buy_v + sell_v).clip(0)
        trade_vol_log = np.log1p(trade_vol)
        df["kyle_lambda"] = (df["ret_1s"].abs() / (trade_vol_log + 1e-6)).astype(np.float32)

        tc_s    = pd.Series(tc).clip(0)
        tc_r10  = tc_s.rolling(10,  min_periods=1).mean()
        tc_r100 = tc_s.rolling(100, min_periods=5).mean()
        df["trade_intensity"] = (tc_r10 / (tc_r100 + 1e-6)).astype(np.float32)

        tv_r10       = trade_vol.rolling(10,  min_periods=1).sum()
        tv_r100_mean = trade_vol.rolling(100, min_periods=10).mean()
        df["vol_momentum"] = (
            np.log1p(tv_r10) / (np.log1p(tv_r100_mean * 10 + 1e-9))
        ).clip(0, 5).fillna(1).astype(np.float32)

        lr_lag  = lr.shift(1)
        cov_rr  = lr.rolling(100, min_periods=20).cov(lr_lag)
        df["roll_spread"] = (2.0 * np.sqrt(np.maximum(-cov_rr, 0.0)) * bps
                             ).clip(0, 20).fillna(0).astype(np.float32)

        df["spread_ma"] = pd.Series(np.abs(spread_bps_arr)).rolling(30, min_periods=5).mean().clip(0, 100).astype(np.float32)

        log_mid_s = pd.Series(np.log(np.where(mid > 0, mid, np.nan)))
        mid_ma = log_mid_s.rolling(200, min_periods=20).mean()
        mid_sd = log_mid_s.rolling(200, min_periods=20).std()
        bb_lower = mid_ma - 2 * mid_sd; bb_upper = mid_ma + 2 * mid_sd
        bb_range = (bb_upper - bb_lower).replace(0, np.nan)
        df["bb_pos"] = ((log_mid_s - bb_lower) / bb_range - 0.5).fillna(0).astype(np.float32)

        w_levels = np.array([1 / (2 ** k) for k in range(N_LEVELS)])
        bid_w = sum(w_levels[k] * np.expm1(bid_sz_log[:, k].clip(0)) for k in range(N_LEVELS))
        ask_w = sum(w_levels[k] * np.expm1(ask_sz_log[:, k].clip(0)) for k in range(N_LEVELS))
        df["depth_imb_w"] = ((bid_w - ask_w) / (bid_w + ask_w + 1e-9)).clip(-1, 1).astype(np.float32)

        buy_r10  = pd.Series(buy_v).clip(0).rolling(10, min_periods=1).sum()
        sell_r10 = pd.Series(sell_v).clip(0).rolling(10, min_periods=1).sum()
        df["aggr_buy"]  = (buy_r10  / (pd.Series(bid0) + 1e-9)).clip(0, 50).astype(np.float32)
        df["aggr_sell"] = (sell_r10 / (pd.Series(ask0) + 1e-9)).clip(0, 50).astype(np.float32)

        df["wmp_change_1s"] = (df["wmp_bps"].diff(10).rolling(5, min_periods=1).mean()
                               ).fillna(0).clip(-20, 20).astype(np.float32)

        df["depth_pressure"] = (pd.Series(log_total_depth).diff(1).rolling(10, min_periods=1).mean()
                                ).fillna(0).clip(-2, 2).astype(np.float32)

        bid0_diff = pd.Series(bid0).diff(1).fillna(0)
        ask0_diff = pd.Series(ask0).diff(1).fillna(0)
        df["cancel_proxy_bid"] = ((-bid0_diff.clip(upper=0)).rolling(10, min_periods=1).mean()).clip(0, 0.5).astype(np.float32)
        df["cancel_proxy_ask"] = ((-ask0_diff.clip(upper=0)).rolling(10, min_periods=1).mean()).clip(0, 0.5).astype(np.float32)

        df["spread_diff"] = (df["roll_spread"] - df["spread_ma"]).clip(-20, 20).astype(np.float32)

        sf = pd.Series(buy_v - sell_v).fillna(0)
        df["acf1_sf_1s"] = (sf.rolling(10, min_periods=4).corr(sf.shift(1))).fillna(0).clip(-1, 1).astype(np.float32)
        sf_sign  = np.sign(sf); prev_sig = sf_sign.shift(1)
        valid_p  = (sf_sign != 0) & (prev_sig != 0)
        df["run_ratio_sf_1s"] = (((sf_sign == prev_sig) & valid_p).astype(np.float32)
                                  .rolling(10, min_periods=1).mean().fillna(0).clip(0, 1)).astype(np.float32)
        df["flip_rate_sf_1s"] = (((sf_sign * prev_sig) < 0).astype(np.float32)
                                  .rolling(10, min_periods=1).mean().fillna(0).clip(0, 1)).astype(np.float32)

        # 时段上下文（与 add_rolling_features 完全一致）
        sec_in_slot = (ts_arr // 1000) % 300
        phase = 2.0 * np.pi * (sec_in_slot / 300.0)
        df["sec_in_slot_sin"]   = np.sin(phase).astype(np.float32)
        df["sec_in_slot_cos"]   = np.cos(phase).astype(np.float32)
        df["is_slot_first_10s"] = (sec_in_slot < 10).astype(np.float32)
        df["is_slot_last_30s"]  = (sec_in_slot >= 270).astype(np.float32)

        # ── 状态特征（与 build_5min_dataset.py 完全一致）─────────────────────
        slot_ms_arr  = (ts_arr // SLOT_MS) * SLOT_MS
        slot_end_arr = slot_ms_arr + SLOT_MS
        ts_float     = ts_arr.astype(np.float64)

        time_remaining = np.clip(
            (slot_end_arr.astype(np.float64) - ts_float) / 1000.0,
            0.0, 300.0
        ).astype(np.float32)
        time_frac = ((300.0 - time_remaining) / 300.0).astype(np.float32)
        pi_frac   = np.pi * time_frac
        df["time_remaining_s"] = time_remaining
        df["time_frac"]        = time_frac
        df["time_sin"]         = np.sin(pi_frac).astype(np.float32)
        df["time_cos"]         = np.cos(pi_frac).astype(np.float32)

        # 每个 tick 用其所在 slot 的 open_price
        open_arr = np.array(
            [self.get_open_price(int(s)) for s in slot_ms_arr],
            dtype=np.float64,
        )
        # fallback（不应触发，万一 slot 未记录）
        nan_mask = np.isnan(open_arr)
        if nan_mask.any():
            for i in np.where(nan_mask)[0]:
                s = int(slot_ms_arr[i])
                first_mid = next((mid[j] for j in range(i + 1) if slot_ms_arr[j] == s), mid[0])
                open_arr[i] = first_mid

        gap = (mid - open_arr).astype(np.float32)
        df["current_gap"]    = gap
        df["gap_abs"]        = np.abs(gap)
        df["gap_sign"]       = np.sign(gap).astype(np.float32)
        df["gap_per_second"] = (gap / (time_remaining + 1.0)).astype(np.float32)

        # gap_normalized（与 build_5min_dataset.py 完全一致）
        # 关键修复：vol_300s 基于完整长期 mid 历史（最多 3000 tick）计算，
        # 而非截断后的 window 窗口，避免行情平静时 std 极小导致 gap_normalized 爆到 ±5
        long_mid_ser = pd.Series(long_mid)
        vol_300s_full = long_mid_ser.diff().abs().rolling(3000, min_periods=100).std().fillna(1.0).clip(lower=0.01)
        # 只取最后 window 个值，与 df 行数对齐
        vol_300s = vol_300s_full.iloc[-T:].values
        expected_move = vol_300s * np.sqrt(time_remaining + 1.0)
        df["gap_normalized"] = (gap / expected_move).clip(-5, 5).astype(np.float32)

        # ── 归一化（apply_norm，与 train_5min_tcn.py 完全一致）────────────────
        if norm_params is not None:
            arr = df[MICRO_FEATURES].values.astype(np.float32)
            arr = np.clip(arr, norm_params["p1"], norm_params["p99"])
            arr = (arr - norm_params["mean"]) / norm_params["std"]
            df[MICRO_FEATURES] = arr

            df["time_remaining_s"] = df["time_remaining_s"] / 300.0
            df["current_gap"]      = df["current_gap"]      / 200.0
            df["gap_abs"]          = df["gap_abs"]           / 200.0
            df["gap_per_second"]   = df["gap_per_second"]    / 10.0

        # ── 组装 [T, 85] ─────────────────────────────────────────────────────
        missing = [c for c in ALL_FEATURES if c not in df.columns]
        if missing:
            print(f"[warn] 缺少特征列: {missing[:5]}")
            for c in missing:
                df[c] = 0.0

        feat_mat = df[ALL_FEATURES].values.astype(np.float32)
        feat_mat = np.nan_to_num(feat_mat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat_mat  # [T, 85]

    @property
    def current_state(self) -> dict:
        if not self.ticks:
            return {}
        last = self.ticks[-1]
        ts   = last["ts"]
        slot = (ts // SLOT_MS) * SLOT_MS
        slot_end = slot + SLOT_MS
        time_remaining = max(0.0, (slot_end - ts) / 1000.0)
        op  = self.get_open_price(slot)
        display_mid = last["mid"]  # 盘口中间价 (best_bid + best_ask) / 2，与训练数据一致
        gap = display_mid - op if not math.isnan(op) else 0.0
        return {
            "ts":               int(time.time() * 1000),  # 实时时间戳，保证每次唯一
            "slot_ms":          slot,
            "open_price":       round(float(op), 2) if not math.isnan(op) else None,
            "mid":              round(float(display_mid), 2),
            "current_gap":      round(float(gap), 2),
            "time_remaining_s": round(float(time_remaining), 1),
            "n_ticks":          len(self.ticks),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_slot(slot_ms: int) -> str:
    import datetime
    t  = datetime.datetime.utcfromtimestamp(slot_ms / 1000)
    t2 = t + datetime.timedelta(minutes=5)
    return f"{t.strftime('%H:%M')}-{t2.strftime('%H:%M')} UTC"


# ─────────────────────────────────────────────────────────────────────────────
# 模型推断
# ─────────────────────────────────────────────────────────────────────────────
class ModelInfer:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[model] 加载 {model_path}  device={self.device}")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model = FiveMinTCNLSTM(
            n_features=ckpt.get("n_features", N_FEATURES),
            n_micro   =ckpt.get("n_micro",    len(MICRO_FEATURES)),
            n_state   =ckpt.get("n_state",    len(STATE_FEATURES)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.temperature = float(ckpt.get("temperature", 1.0))
        self.norm_params = {
            "mean": ckpt["norm_mean"],
            "std":  ckpt["norm_std"],
            "p1":   ckpt["norm_p1"],
            "p99":  ckpt["norm_p99"],
        }
        fw_info = "(GateNetwork 动态融合)"
        try:
            fw_info = f"(GateNetwork 动态融合，新架构)"
        except Exception:
            pass
        print(f"[model] T={self.temperature:.4f}  {fw_info}")

    @torch.no_grad()
    def infer(self, feat_mat: np.ndarray) -> tuple[float, float]:
        """返回 (prob_up_calibrated, prob_up_raw)"""
        x = torch.from_numpy(feat_mat[None]).to(self.device)
        out = self.model(x)
        # 新架构返回3元组 (prob_settlement, prob_dir_30s, prob_dir_3s)
        if isinstance(out, tuple):
            prob_raw = out[0].squeeze().item()
            self._last_prob_30s = out[1].squeeze().item()
            self._last_prob_3s  = out[2].squeeze().item()
        else:
            prob_raw = out.squeeze().item()
            self._last_prob_30s = None
            self._last_prob_3s  = None
        logit    = math.log(max(prob_raw, 1e-7) / max(1 - prob_raw, 1e-7))
        prob_cal = 1.0 / (1.0 + math.exp(-logit / self.temperature))
        return round(prob_cal, 4), round(prob_raw, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 推理主循环
# ─────────────────────────────────────────────────────────────────────────────
PROB_EMA_ALPHA = 0.25   # EMA 平滑系数：值越小越平滑，0.25 = 约4个tick的半衰期

class InferenceEngine:
    def __init__(self, model_path: str):
        self.infer_model = ModelInfer(model_path)
        self.feat_engine = FeatureEngine()
        self.tick_agg    = TickAggregator()

        self.history:       Deque[dict] = collections.deque(maxlen=HISTORY_LEN)
        self.last_infer_ms: int  = 0
        self.last_result:   dict = {}
        self.ws_clients:    set  = set()

        # EMA 平滑状态（输出侧，不影响特征）
        self._ema_prob:     Optional[float] = None   # 当前 EMA 值
        self._ema_slot_ms:  int = -1                 # EMA 所属 slot，换局即重置

    def on_binance_msg(self, raw_msg: str):
        try:
            msg = json.loads(raw_msg)
        except Exception:
            return

        # 直连流（/ws/stream1/stream2）消息直接就是事件本体，无外层 stream/data 包装
        etype = msg.get("e", "")

        if etype == "depthUpdate":
            self.tick_agg.on_depth_update(msg)
        elif etype == "trade":
            self.tick_agg.on_trade(msg)

        while self.tick_agg.completed:
            tick = self.tick_agg.completed.popleft()
            self.feat_engine.push(tick)

        now = int(time.time() * 1000)
        if now - self.last_infer_ms >= INFER_EVERY_MS:
            self._do_infer(now)
            self.last_infer_ms = now

    def _do_infer(self, now_ms: int):
        state   = self.feat_engine.current_state
        n_ticks = len(self.feat_engine.ticks)
        model_ready = (
            n_ticks >= MIN_TICKS
            and self.tick_agg._ob_initialized
            and state.get("open_price") is not None
        )

        prob_up = prob_up_raw = None

        if model_ready:
            try:
                feat = self.feat_engine.build_feature_window(
                    window=min(WINDOW, n_ticks),
                    norm_params=self.infer_model.norm_params,
                )
                if feat is not None:
                    prob_up, prob_up_raw = self.infer_model.infer(feat)
            except Exception as e:
                import traceback
                print(f"[infer error] {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        # 取辅助方向概率（新架构）
        prob_dir_3s  = getattr(self.infer_model, "_last_prob_3s",  None)
        prob_dir_30s = getattr(self.infer_model, "_last_prob_30s", None)

        result = {
            **state,
            "trade_price": round(self.tick_agg.last_trade_price, 2)
                           if not math.isnan(self.tick_agg.last_trade_price) else state.get("mid"),
            "prob_up":      prob_up,
            "prob_up_raw":  prob_up_raw,
            "prob_dir_3s":  round(prob_dir_3s,  4) if prob_dir_3s  is not None else None,
            "prob_dir_30s": round(prob_dir_30s, 4) if prob_dir_30s is not None else None,
            "temperature":  self.infer_model.temperature,
            "model_ready":  model_ready,
            "history":      list(self.history),
        }

        if prob_up is not None:
            self.history.append({
                "ts":      now_ms,
                "mid":     state.get("mid"),
                "prob_up": prob_up,
                "gap":     state.get("current_gap"),
            })
            tr  = state.get("time_remaining_s", "?")
            gap = state.get("current_gap", 0.0)
            op  = state.get("open_price", "?")
            dir3s_str  = f"  dir3s={prob_dir_3s:.3f}"  if prob_dir_3s  is not None else ""
            dir30s_str = f"  dir30s={prob_dir_30s:.3f}" if prob_dir_30s is not None else ""
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"mid={state.get('mid','?')}  open={op}  "
                f"gap={gap:+.1f}  time_left={tr}s  "
                f"prob_up={prob_up:.4f}(raw={prob_up_raw:.4f})"
                f"{dir3s_str}{dir30s_str}  "
                f"ticks={n_ticks}",
                flush=True,
            )

        self.last_result = result

        if self.ws_clients:
            payload = json.dumps(
                {k: v for k, v in result.items() if k != "history"}
                | {"history": list(self.history)[-60:]}
            )
            asyncio.create_task(_broadcast(self.ws_clients, payload))

    async def ws_handler(self, websocket):
        self.ws_clients.add(websocket)
        print(f"[ws] 前端连接 (共 {len(self.ws_clients)} 客户端)")
        try:
            if self.last_result:
                await websocket.send(json.dumps(self.last_result))
            async for _ in websocket:
                pass
        except Exception:
            pass
        finally:
            self.ws_clients.discard(websocket)
            print(f"[ws] 前端断开 (共 {len(self.ws_clients)} 客户端)")


async def _broadcast(clients: set, payload: str):
    for ws in list(clients):
        try:
            await ws.send(payload)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Binance REST 快照
# ─────────────────────────────────────────────────────────────────────────────
async def _init_snapshot(agg: TickAggregator):
    """
    标准 Binance 增量订单簿同步流程：
      WS 已连接（缓冲区开始积累）→ 请求 REST /depth →
      丢弃 u <= lastUpdateId 的 pending → 应用快照 + 回放
    """
    if not HAS_AIOHTTP:
        print("[warn] aiohttp 未安装，跳过快照（盘口精度下降）")
        agg._syncing = False
        agg._ob_initialized = True
        return

    url = f"{BINANCE_REST}/depth?symbol={SYMBOL}&limit=1000"
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
            agg.apply_snapshot(data)
            return
        except Exception as e:
            print(f"[snapshot] 获取失败 (attempt {attempt+1}/3): {e}")
            await asyncio.sleep(2)

    print("[snapshot] 快照获取失败，强制继续（精度下降）")
    agg._syncing = False
    agg._ob_initialized = True


# ─────────────────────────────────────────────────────────────────────────────
# Binance WebSocket 连接
# ─────────────────────────────────────────────────────────────────────────────
async def binance_feed(engine: InferenceEngine):
    # 直连组合流：/ws/<stream1>/<stream2>  — 无外层封装，延迟更低
    url = f"{BINANCE_WS_URL}/{SYMBOL_LOWER}@depth@100ms/{SYMBOL_LOWER}@trade"

    while True:
        try:
            print(f"[binance] 连接 {url}")
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                max_size=10 * 1024 * 1024,
            ) as ws:
                # WS 建立后立即获取快照（此时 pending 缓冲已开始积累）
                print("[binance] WS 已连接，请求 REST 快照...")
                await _init_snapshot(engine.tick_agg)

                async for raw_msg in ws:
                    engine.on_binance_msg(raw_msg)

        except Exception as e:
            print(f"[binance] 断线: {e}  5秒后重连...", file=sys.stderr)
            # 重置同步状态
            engine.tick_agg._syncing = True
            engine.tick_agg._ob_initialized = False
            engine.tick_agg._pending_updates.clear()
            await asyncio.sleep(5)


# ─────────────────────────────────────────────────────────────────────────────
# 前端 WebSocket 服务
# ─────────────────────────────────────────────────────────────────────────────
async def run_ws_server(engine: InferenceEngine, port: int):
    async with websockets.serve(engine.ws_handler, "127.0.0.1", port) as server:
        print(f"[ws-server] 前端 WebSocket 监听 ws://127.0.0.1:{port}")
        await server.serve_forever()


# ─────────────────────────────────────────────────────────────────────────────
# 内置 HTTP 服务器（提供 dashboard.html）
# ─────────────────────────────────────────────────────────────────────────────
async def run_http_server(html_path: Path, http_port: int):
    """用 asyncio 异步方式运行一个极简 HTTP 服务，专门提供 dashboard.html
    同时提供 /api/pm-market 代理，解决浏览器 CORS 限制"""
    import http.server
    import threading
    import urllib.request

    html_dir = str(html_path.parent)
    html_file = html_path.name

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=html_dir, **kw)

        def do_GET(self):
            # 代理 /api/pm-market?ts=xxx → gamma-api.polymarket.com
            if self.path.startswith('/api/pm-market'):
                self._proxy_pm()
                return
            # 根路径 / 直接返回 dashboard.html
            if self.path in ('/', ''):
                self.path = '/' + html_file
            super().do_GET()

        def _proxy_pm(self):
            """服务端代理 Polymarket gamma-api，避免浏览器 CORS"""
            from urllib.parse import urlparse, parse_qs
            qs  = parse_qs(urlparse(self.path).query)
            ts  = qs.get('ts', [None])[0]
            if not ts:
                self.send_error(400, 'missing ts')
                return
            upstream = f'https://gamma-api.polymarket.com/markets?slug=btc-updown-5m-{ts}'
            try:
                req = urllib.request.Request(
                    upstream,
                    headers={'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'},
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    body = resp.read()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_error(502, str(e))

        def log_message(self, fmt, *args):
            pass  # 静默日志

    def _serve():
        # 尝试多个地址绑定
        for host in ('127.0.0.1', '0.0.0.0', ''):
            try:
                server = http.server.HTTPServer((host, http_port), Handler)
                print(f"[http] dashboard 已启动 → http://127.0.0.1:{http_port}  (bind={host!r})")
                server.serve_forever()
                return
            except OSError as e:
                print(f"[http] 绑定 {host!r}:{http_port} 失败: {e}，尝试下一个...")
        print(f"[http] 警告: HTTP服务启动失败，请直接用浏览器打开 YUCE/dashboard.html")

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    # 保持协程活着
    while True:
        await asyncio.sleep(3600)


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
async def main_async(args):
    engine = InferenceEngine(args.model)
    tasks  = [binance_feed(engine)]
    if not args.no_ws and HAS_WS:
        tasks.append(run_ws_server(engine, args.port))

    # 内置 HTTP 服务：提供 dashboard.html
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        tasks.append(run_http_server(html_path, args.http_port))

    print(f"\n{'='*60}")
    print(f"BTC 5分钟方向实时推理（真实币安 WSS）")
    print(f"  模型:   {args.model}")
    print(f"  设备:   {engine.infer_model.device}")
    print(f"  WS:     ws://localhost:{args.port}")
    print(f"  Dashboard: http://localhost:{args.http_port}")
    print(f"  推理:   每 {INFER_EVERY_MS}ms 一次，最少 {MIN_TICKS} tick")
    print(f"  特征:   {N_FEATURES} 维（微结构{len(MICRO_FEATURES)} + 状态{len(STATE_FEATURES)}）")
    print(f"{'='*60}\n")

    await asyncio.gather(*tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     default=DEFAULT_MODEL)
    ap.add_argument("--port",      type=int, default=WS_PORT)
    ap.add_argument("--http-port", type=int, default=HTTP_PORT)
    ap.add_argument("--no-ws",     action="store_true")
    args = ap.parse_args()

    if not HAS_WS:
        print("[error] 请安装 websockets：pip install websockets aiohttp")
        sys.exit(1)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[exit] 用户中止")


if __name__ == "__main__":
    main()
