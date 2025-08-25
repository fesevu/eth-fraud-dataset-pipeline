#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM daily → weekly → monthly (sync with GNN windows) — ultra low‑mem (≤12 GB).

Вход:
  /mnt/d/new_Fraud/dataset/final/LSTM/parquet/*.parquet
  /mnt/d/new_Fraud/dataset/final/gnn_dataset/labels/targets_global.parquet
  /mnt/d/new_Fraud/dataset/final/gnn_dataset/mapping/address_id_map_labels.parquet
  /mnt/d/new_Fraud/dataset/final/gnn_dataset/meta/week_window_meta.parquet
  /mnt/d/new_Fraud/dataset/final/gnn_dataset/meta/month_window_meta.parquet

Выход (отдельная папка, НЕ gnn_dataset):
  /mnt/d/new_Fraud/dataset/final/lstm_dataset/
    ├─ daily_filtered.parquet
    ├─ weekly.parquet
    ├─ monthly.parquet
    └─ README.md
"""

from __future__ import annotations
from pathlib import Path
import os
import polars as pl

# ---- 12 GB friendly ----
os.environ["POLARS_MAX_THREADS"] = os.environ.get("POLARS_MAX_THREADS", "2")
ROW_GROUP = 128_000  # маленькие row-groups снижают пиковую память при записи

BASE = Path("/mnt/d/new_Fraud/dataset/final")
SRC_LSTM_DAILY = BASE / "LSTM" / "parquet"         # твои дневные parquet-файлы

# общий набор для GNN (labels/meta)
DS_ROOT = BASE / "gnn_dataset"
# отдельный корень LSTM-выходов
OUT_DIR = BASE / "lstm_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_FP = DS_ROOT / "labels" / "targets_global.parquet"
MAP_FP = DS_ROOT / "mapping" / "address_id_map_labels.parquet"
WEEK_META = DS_ROOT / "meta" / "week_window_meta.parquet"
MONTH_META = DS_ROOT / "meta" / "month_window_meta.parquet"

DAILY_OUT = OUT_DIR / "daily_filtered.parquet"
WEEKLY_OUT = OUT_DIR / "weekly.parquet"
MONTHLY_OUT = OUT_DIR / "monthly.parquet"
README_FP = OUT_DIR / "README.md"

TMP_WEEK_DIR = OUT_DIR / "_tmp_week_parts"
TMP_MONTH_DIR = OUT_DIR / "_tmp_month_parts"
TMP_WEEK_DIR.mkdir(parents=True, exist_ok=True)
TMP_MONTH_DIR.mkdir(parents=True, exist_ok=True)


def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect()


# ---- поля ----
ETH_NUMS = [
    "eth_sent_sum", "eth_recv_sum", "eth_net_flow",
    "tx_fee_eth_sum", "internal_out_value_eth_sum", "internal_in_value_eth_sum",
]
INT_COUNTS = [
    "normal_sent_cnt", "normal_recv_cnt", "normal_total_cnt", "normal_failed_cnt",
    "normal_to_contract_cnt", "normal_to_eoa_cnt",
    "gas_used_sum",
    "erc20_sent_cnt", "erc20_recv_cnt", "erc20_total_cnt",
    "erc20_unique_tokens_sent", "erc20_unique_tokens_recv",
    "internal_out_cnt", "internal_in_cnt",
    "uniq_peers_cnt", "uniq_contract_peers_cnt", "uniq_eoa_peers_cnt",
    "sessions_cnt",
    "active_span_min",
    "burst_max_tx_5m",  # MAX over window
]
MAX_ONLY = {"burst_max_tx_5m"}
SUM_ONLY = (set(INT_COUNTS) - MAX_ONLY) | set(ETH_NUMS)

# ============== 1) targets + mapping ==============
if not LABELS_FP.exists() or not MAP_FP.exists():
    raise FileNotFoundError(
        "Не найдены labels/mapping в gnn_dataset. Сначала собери GNN gnn_dataset.")
targets = pl.read_parquet(LABELS_FP).select(
    ["node_id", "address"]).with_columns(pl.col("address").str.to_lowercase())
addr_map = pl.read_parquet(MAP_FP).with_columns(
    pl.col("address").str.to_lowercase()).unique(subset=["address"])

# ============== 2) daily → filtered + node_id + week/month (streaming) ==============
if not any(SRC_LSTM_DAILY.glob("*.parquet")):
    raise FileNotFoundError(
        f"Не найдены входные daily parquet в {SRC_LSTM_DAILY}")

# Берём только реально существующие колонки, не дергая полную схему (без предупреждения)
src_scan = pl.scan_parquet(str(SRC_LSTM_DAILY / "*.parquet"))
src_schema = src_scan.collect_schema()
needed = ["address", "day"] + \
    [c for c in (INT_COUNTS + ETH_NUMS) if c in src_schema]
lf_daily = (
    src_scan
    .select(needed)
    .with_columns(pl.col("address").str.to_lowercase())
    # только целевые адреса; добавит node_id
    .join(addr_map.lazy(), on="address", how="inner")
    .with_columns([
        pl.col("day").cast(pl.Date),
        pl.col("day").dt.strftime("%G-W%V").alias("week"),
        pl.col("day").dt.strftime("%Y-%m").alias("month"),
    ])
    .select(["node_id", "address", "day", "week", "month"] + [c for c in needed if c not in {"address", "day"}])
)
lf_daily.sink_parquet(str(DAILY_OUT), compression="zstd", statistics=True)
print(f"[✓] daily_filtered → {DAILY_OUT}")

# ============== 3) weekly — ПО ОКНАМ (низкая память) ==============
if not WEEK_META.exists():
    raise FileNotFoundError(WEEK_META)
weeks_list = pl.read_parquet(WEEK_META).select("week").to_series().to_list()

# агрегации


def sum_int(c): return pl.col(c).cast(
    pl.Int64, strict=False).fill_null(0).sum().alias(c)


def max_int(c): return pl.col(c).cast(
    pl.Int64, strict=False).fill_null(0).max().alias(c)


def sum_dec_str(c): return pl.col(c).cast(pl.Decimal(38, 9),
                                          strict=False).fill_null(0).sum().cast(pl.Utf8).alias(c)


weekly_aggs = []
for c in SUM_ONLY:
    if c in ETH_NUMS:
        weekly_aggs.append(sum_dec_str(c))
    else:
        weekly_aggs.append(sum_int(c))
for c in MAX_ONLY:
    weekly_aggs.append(max_int(c))

# считаем каждую неделю отдельно и пишем part-файлы
for i, w in enumerate(weeks_list, 1):
    part_fp = TMP_WEEK_DIR / f"week={w}.parquet"
    if part_fp.exists():
        if i % 50 == 0:
            print(f"[weekly] skip {i}/{len(weeks_list)}")
        continue
    lf_part = (
        pl.scan_parquet(str(DAILY_OUT))
          .filter(pl.col("week") == w)
          .group_by(["node_id", "week"])
          .agg(weekly_aggs)
          .select(["node_id", "week"] + sorted(SUM_ONLY | MAX_ONLY))
    )
    # маленький фрейм → пишем напрямую
    collect_streaming(lf_part).write_parquet(
        part_fp, compression="zstd", statistics=True, row_group_size=ROW_GROUP)
    if i % 50 == 0:
        print(f"[weekly] {i}/{len(weeks_list)}")

# объединяем все weekly-парты в один файл стримингом
parts = sorted(TMP_WEEK_DIR.glob("week=*.parquet"))
if parts:
    pl.scan_parquet([str(p) for p in parts]).sink_parquet(
        str(WEEKLY_OUT), compression="zstd", statistics=True)
print(f"[✓] weekly → {WEEKLY_OUT}")

# ============== 4) month — ИЗ НЕДЕЛЬ (низкая память) ==============
if not MONTH_META.exists():
    raise FileNotFoundError(MONTH_META)

# компактная таблица соответствия (week → month) из уже сохранённого daily_filtered
week_month_map = (
    pl.scan_parquet(str(DAILY_OUT))
      .select(["week", "month"])
      .unique()
      .collect()     # это маленькая таблица (по числу недель)
)

months_list = pl.read_parquet(MONTH_META).select("month").to_series().to_list()

# При агрегации по месяцам:


def sum_int_m(c): return pl.col(c).cast(
    pl.Int64, strict=False).fill_null(0).sum().alias(c)


def sum_dec_str_m(c): return pl.col(c).cast(pl.Utf8).cast(
    pl.Decimal(38, 9), strict=False).fill_null(0).sum().cast(pl.Utf8).alias(c)


def max_int_m(c): return pl.col(c).cast(
    pl.Int64, strict=False).fill_null(0).max().alias(c)


monthly_aggs = []
for c in SUM_ONLY:
    if c in ETH_NUMS:
        # weekly-значения строковые → обратно в Decimal → sum → строка
        monthly_aggs.append(sum_dec_str_m(c))
    else:
        monthly_aggs.append(sum_int_m(c))
for c in MAX_ONLY:
    monthly_aggs.append(max_int_m(c))

for i, m in enumerate(months_list, 1):
    part_fp = TMP_MONTH_DIR / f"month={m}.parquet"
    if part_fp.exists():
        if i % 25 == 0:
            print(f"[monthly] skip {i}/{len(months_list)}")
        continue

    # недели, входящие в месяц m
    w_in_m = week_month_map.filter(pl.col("month") == m).select(
        "week").to_series().to_list()
    if not w_in_m:
        continue

    lf_m = (
        pl.scan_parquet(str(WEEKLY_OUT))
          .filter(pl.col("week").is_in(w_in_m))
          .group_by("node_id")
          .agg(monthly_aggs)
          .with_columns(pl.lit(m).alias("month"))
          .select(["node_id", "month"] + sorted(SUM_ONLY | MAX_ONLY))
    )
    collect_streaming(lf_m).write_parquet(
        part_fp, compression="zstd", statistics=True, row_group_size=ROW_GROUP)
    if i % 25 == 0:
        print(f"[monthly] {i}/{len(months_list)}")

# объединяем все monthly-парты в один файл
parts_m = sorted(TMP_MONTH_DIR.glob("month=*.parquet"))
if parts_m:
    pl.scan_parquet([str(p) for p in parts_m]).sink_parquet(
        str(MONTHLY_OUT), compression="zstd", statistics=True)
print(f"[✓] monthly → {MONTHLY_OUT}")

# ============== 5) README.md ==============
README_FP.write_text(
    "# 📊 Fraud LSTM Dataset\n\n"
    "Датасет для обучения LSTM, синхронизирован с GNN (`gnn_dataset`) по адресам (`node_id`) и окнам (`week`, `month`).\n\n"
    "## Файлы\n"
    "- `daily_filtered.parquet` — дневные фичи только для целевых адресов, с колонками `node_id`, `address`, `day`, `week`, `month`.\n"
    "- `weekly.parquet` — агрегаты по ISO‑неделям (YYYY-Www).\n"
    "- `monthly.parquet` — агрегаты по месяцам (YYYY-MM), собраны **из недель**.\n\n"
    "## Агрегации\n"
    "- Все счётчики суммируются.\n"
    "- ETH‑поля суммируются как Decimal(38,9) и сохраняются строкой (без потери точности).\n"
    "- `burst_max_tx_5m` — берём максимум в окне.\n\n"
    "## Синхронизация\n"
    "- Совпадающие `node_id` с `gnn_dataset/labels/targets_global.parquet`.\n"
    "- Совпадающие окна с `gnn_dataset/meta/week_window_meta.parquet` и `month_window_meta.parquet`.\n",
    encoding="utf-8"
)
print(f"[✓] README → {README_FP}")

print(f"\n[✓] LSTM dataset ready → {OUT_DIR}\n"
      f"    weekly parts:  {TMP_WEEK_DIR}\n"
      f"    monthly parts: {TMP_MONTH_DIR}\n")
