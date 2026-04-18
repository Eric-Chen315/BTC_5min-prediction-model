"""
build_5min_dataset.py  ——  构建 Polymarket BTC 5分钟方向预测数据集
=======================================================================

逻辑：
  BTC 5分钟预测市场（Polymarket）规则：
    - 开盘价 = 每5分钟整点时刻 Binance BTC 的 mid（K线 open）
    - 结算价 = 5分钟后整点时刻的 mid（K线 close）
    - label_5min_up = 1 if close > open else 0

  每个5分钟"局"的每个 tick 都生成一个训练样本，额外包含：
    - time_remaining_s   : 距离结算的剩余秒数（0.1 ~ 300）
    - current_gap        : mid - open_price（当前领先/落后）
    - gap_abs            : abs(current_gap)
    - gap_sign           : sign(current_gap)，+1=UP领先，-1=DOWN领先
    - gap_zscore         : current_gap / rolling_vol_300s（用波动率标准化）
    - gap_per_second     : current_gap / (time_remaining_s + 1)（翻转所需速度）
    - gap_normalized     : gap / (vol * sqrt(time_remaining))（物理意义最强）
    - time_frac          : (300 - time_remaining_s) / 300（已消耗时间比例）
    - time_sin/cos       : 时间编码（周期性特征）

  合并以上 9 个状态特征 + 原有 76 个微结构特征 = 85 个输入特征

  label_5min_up：最终结果（close > open），整局所有 tick 共享同一个 label

用法：
  python build_5min_dataset.py --start 2026-03-01 --end 2026-03-30
  python build_5min_dataset.py --start 2026-03-25 --end 2026-03-30 --out DATE/5min_dataset
"""

from __future__ import annotations

import argparse
import warnings
from datetime import date as _date, timedelta as _td
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 特征列（与 train_tcn_lstm.py 一致）
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

# 原始 76 个微结构特征
MICRO_FEATURES = L2_COLS + TRADE_COLS + DERIVED_COLS   # 76列

# 新增 9 个"局面状态"特征（核心创新）
STATE_FEATURES = [
    "time_remaining_s",   # 剩余秒数（0.1 ~ 300）
    "time_frac",          # 已消耗时间比（0 ~ 1）
    "time_sin",           # sin(π * time_frac)
    "time_cos",           # cos(π * time_frac)
    "current_gap",        # mid - open_price（正=UP领先）
    "gap_abs",            # abs(current_gap)
    "gap_sign",           # sign(current_gap)
    "gap_per_second",     # gap / (time_remaining + 1)，翻转所需速度
    "gap_normalized",     # gap / (vol_300s * sqrt(time_remaining+1))，物理标准化
]

# 全部输入特征：76 + 9 = 85
ALL_5MIN_FEATURES = MICRO_FEATURES + STATE_FEATURES
N_5MIN_FEATURES   = len(ALL_5MIN_FEATURES)   # 85

print(f"[config] 微结构特征={len(MICRO_FEATURES)}  状态特征={len(STATE_FEATURES)}  总计={N_5MIN_FEATURES}")


# ─────────────────────────────────────────────────────────────────────────────
# 核心处理：一天的 parquet → 5分钟局数据集
# ─────────────────────────────────────────────────────────────────────────────
def process_one_day(src_path: Path) -> pd.DataFrame | None:
    """
    读取一天的 l2_dataset parquet，切割成5分钟局，
    为每个 tick 添加状态特征 + label_5min_up。
    """
    if not src_path.exists():
        print(f"  [skip] {src_path.name} 不存在")
        return None

    df = pd.read_parquet(src_path)

    # 确认必要列存在
    missing = [c for c in MICRO_FEATURES if c not in df.columns]
    if missing:
        print(f"  [warn] {src_path.name} 缺少特征列: {missing[:5]}...")
        return None

    df = df.sort_values("event_time").reset_index(drop=True)

    # ── 切割5分钟slot ──────────────────────────────────────────────────────
    # slot_id = Unix ms // 300000（每300秒一个整点slot）
    df["slot_ms"] = (df["event_time"] // 300_000) * 300_000
    slot_end_ms   = df["slot_ms"] + 300_000

    # time_remaining（浮点秒，0.1 ~ 300）
    df["time_remaining_s"] = (slot_end_ms - df["event_time"]) / 1000.0
    df["time_remaining_s"] = df["time_remaining_s"].clip(0.0, 300.0).astype(np.float32)

    # time_frac：已消耗时间比（0=刚开始，1=快结束）
    df["time_frac"] = ((300.0 - df["time_remaining_s"]) / 300.0).astype(np.float32)

    # 时间编码（帮助模型感知"早期"vs"晚期"的非线性差异）
    pi_frac = np.pi * df["time_frac"].values
    df["time_sin"] = np.sin(pi_frac).astype(np.float32)
    df["time_cos"] = np.cos(pi_frac).astype(np.float32)

    # ── 每局的 open_price = slot 第一个 tick 的 mid ──────────────────────
    slot_open = (
        df.groupby("slot_ms")["mid"]
          .first()
          .rename("open_price")
    )
    df = df.join(slot_open, on="slot_ms")

    # ── 每局的 close_price = slot 最后一个 tick 的 mid ───────────────────
    slot_close = (
        df.groupby("slot_ms")["mid"]
          .last()
          .rename("close_price")
    )
    df = df.join(slot_close, on="slot_ms")

    # ── 关键特征：current_gap ─────────────────────────────────────────────
    # 物理含义：当前 mid 比开盘高多少 USDT
    #   > 0 → UP 目前领先（需要 DOWN 方向翻转才能逆转）
    #   < 0 → DOWN 目前领先
    df["current_gap"]  = (df["mid"] - df["open_price"]).astype(np.float32)
    df["gap_abs"]      = df["current_gap"].abs().astype(np.float32)
    df["gap_sign"]     = np.sign(df["current_gap"].values).astype(np.float32)

    # gap_per_second：翻转差价所需的平均速度（USDT/s）
    # 剩余时间内需要每秒移动多少才能翻转
    df["gap_per_second"] = (
        df["current_gap"] / (df["time_remaining_s"] + 1.0)
    ).astype(np.float32)

    # gap_normalized：用波动率标准化（最强物理意义特征）
    # ≈ 用过去5分钟的实现波动率 σ 标准化差价
    # 计算 rolling_std_300s（过去3000tick = 300s）
    vol_300s = df["mid"].diff().abs().rolling(3000, min_periods=100).std().fillna(1.0)
    vol_300s = vol_300s.clip(lower=0.01)   # 防止除零
    expected_move = vol_300s * np.sqrt(df["time_remaining_s"].values + 1.0)
    df["gap_normalized"] = (df["current_gap"] / expected_move).clip(-5, 5).astype(np.float32)

    # ── label_5min_up：整局共享结果 ──────────────────────────────────────
    # 1 = close > open（UP方向赢），0 = DOWN方向赢
    # 平局（close == open）归为0（DOWN赢 or 无效局）
    df["label_5min_up"] = (df["close_price"] > df["open_price"]).astype(np.int8)

    # ── 多 horizon 方向软标签（UP=1.0 / FLAT=0.5 / DOWN=0.0）────────────
    # slot 末尾超出范围的 tick 保留 0.5（平盘处理，不丢样本）
    def _soft_dir_label(g: pd.DataFrame, n_ticks: int) -> pd.Series:
        future_mid = g["mid"].shift(-n_ticks)
        lbl = pd.Series(0.5, index=g.index, dtype=np.float32)
        lbl[future_mid > g["mid"]] = 1.0
        lbl[future_mid < g["mid"]] = 0.0
        return lbl

    df["label_dir_3s"]  = df.groupby("slot_ms", group_keys=False).apply(
        lambda g: _soft_dir_label(g, 30))   # 3s  = 30 tick
    df["label_dir_30s"] = df.groupby("slot_ms", group_keys=False).apply(
        lambda g: _soft_dir_label(g, 300))  # 30s = 300 tick

    # ── dir_weight 已移除 ────────────────────────────────────────────────
    # 原本用 exp(-gap²/2σ²) 手工控制短期任务权重
    # 现改为模型内部可学习门控（GateNetwork），由模型自己发现何时信微结构
    # 训练时直接传 gap_abs / time_remaining 给门控，不再需要预计算权重列

    # ── 过滤无效局 ────────────────────────────────────────────────────────
    # 1. tick数太少的局（< 100 tick = 10秒）可能是数据缺失
    slot_cnt = df.groupby("slot_ms").size()
    valid_slots = slot_cnt[slot_cnt >= 100].index
    n_before = len(df)
    df = df[df["slot_ms"].isin(valid_slots)].copy()
    n_slots_total = df["slot_ms"].nunique()

    # ── 清理中间列 ────────────────────────────────────────────────────────
    df = df.drop(columns=["open_price", "close_price", "slot_ms"], errors="ignore")

    # ── 最终列检验 ────────────────────────────────────────────────────────
    missing_final = [c for c in ALL_5MIN_FEATURES if c not in df.columns]
    if missing_final:
        print(f"  [warn] 最终缺少列: {missing_final}")

    # 确保特征列 float32
    for c in STATE_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype(np.float32)

    up_rate = df["label_5min_up"].mean()
    print(f"  ✓ {src_path.stem}: {n_slots_total}局  {len(df):,}行  "
          f"UP率={up_rate:.1%}  "
          f"gap均值={df['current_gap'].mean():.1f}  "
          f"gap_norm p95={df['gap_normalized'].abs().quantile(0.95):.2f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",  default="2026-03-01")
    ap.add_argument("--end",    default="2026-03-30")
    ap.add_argument("--src",    default=str(Path(__file__).parent.parent / "DATE/l2_dataset"),   help="l2_dataset 目录")
    ap.add_argument("--out",    default=str(Path(__file__).parent.parent / "DATE/5min_dataset"), help="输出目录")
    ap.add_argument("--force",  action="store_true", help="强制重建所有已存在的文件")
    args = ap.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    s = _date.fromisoformat(args.start)
    e = _date.fromisoformat(args.end)
    dates = [(s + _td(days=i)).isoformat() for i in range((e - s).days + 1)]

    print(f"\n{'='*60}")
    print(f"构建 5分钟方向预测数据集")
    print(f"  日期范围: {args.start} ~ {args.end}  共{len(dates)}天")
    print(f"  输入: {src_dir}")
    print(f"  输出: {out_dir}")
    print(f"  特征数: {N_5MIN_FEATURES}（微结构76 + 状态9）")
    print(f"{'='*60}\n")

    total_slots = 0
    total_rows  = 0

    for d in dates:
        src_path = src_dir / f"l2_{d}.parquet"
        out_path = out_dir / f"5min_{d}.parquet"

        if out_path.exists():
            # 检查是否需要重建（源文件更新了 或 --force）
            if args.force:
                print(f"  [force]  {d} 强制重建...")
            elif src_path.exists() and src_path.stat().st_mtime > out_path.stat().st_mtime:
                print(f"  [rebuild] {d} 源文件已更新，重建...")
            else:
                print(f"  [skip]  {d} 已存在")
                continue

        result = process_one_day(src_path)
        if result is None:
            continue

        # 只保存需要的列（节省存储）
        save_cols = ALL_5MIN_FEATURES + ["event_time", "date", "hour", "mid",
                                          "label_5min_up", "label_dir_3s", "label_dir_30s"]
        save_cols = [c for c in save_cols if c in result.columns]
        result[save_cols].to_parquet(out_path, index=False, compression="zstd")

        total_slots += result["slot_ms"].nunique() if "slot_ms" in result.columns else 0
        total_rows  += len(result)

    print(f"\n{'='*60}")
    print(f"✅ 完成！总计约 {total_rows:,} 行")
    print(f"   输出目录: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
