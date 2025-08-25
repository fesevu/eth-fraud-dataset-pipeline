#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Единый датасет для GNN (внутреннее и публичное использование).
Память: ≤ 24 ГБ за счёт итеративной обработки окон и стриминга.

Выход: /<BASE>/gnn_dataset/
  edges_all/edges.parquet
  edges_by_month/month=YYYY-MM/edges.parquet
  edges_by_week/week=YYYY-Www/edges.parquet
  meta/month_window_meta.parquet, meta/week_window_meta.parquet
  labels/targets_global.parquet, mapping/address_id_map_labels.parquet
  targets/month_targets.parquet, targets/week_targets.parquet
  README.md
"""

from __future__ import annotations
from pathlib import Path
import os
import polars as pl

# --------- лимиты для RAM ---------
os.environ.setdefault("POLARS_MAX_THREADS", "4")
ROW_GROUP = 256_000

BASE = Path("/mnt/d/new_Fraud/dataset/final")

SRC_TX_DIR = BASE / "GNN" / "parquet"
LABELS_CSV = BASE / "addr_labels_balanced.csv"

DS_ROOT = BASE / "gnn_dataset"
EDGES_ALL_FP = DS_ROOT / "edges_all" / "edges.parquet"
MONTH_DIR = DS_ROOT / "edges_by_month"
WEEK_DIR = DS_ROOT / "edges_by_week"
META_DIR = DS_ROOT / "meta"
LABELS_DIR = DS_ROOT / "labels"
MAP_DIR = DS_ROOT / "mapping"
TARGETS_DIR = DS_ROOT / "targets"
README_FP = DS_ROOT / "README.md"

for p in [EDGES_ALL_FP.parent, MONTH_DIR, WEEK_DIR, META_DIR, LABELS_DIR, MAP_DIR, TARGETS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

HASH_SEEDS = dict(seed=20250823, seed_1=17, seed_2=31, seed_3=73)


def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect()

# -------------------- 1) labels & mapping --------------------


def build_labels_and_mapping():
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Не найден labels CSV: {LABELS_CSV}")
    lab = pl.read_csv(LABELS_CSV).with_columns(
        pl.col("address").str.to_lowercase().alias("address_lc")
    )
    labels = (
        lab.select([
            "address", "address_lc",
            pl.col("is_scam").cast(pl.Int8),
            pl.col("is_contract").cast(pl.Int8, strict=False),
        ])
        .with_columns([
            pl.col("address_lc").hash(
                **HASH_SEEDS).cast(pl.UInt64).alias("node_id")
        ])
        .unique(subset=["node_id"])
    )
    targets_global = labels.select(
        ["node_id", "is_scam", "is_contract", "address"])
    targets_global.write_parquet(
        LABELS_DIR / "targets_global.parquet", compression="zstd", statistics=True)
    labels.select(["address", "node_id"]).write_parquet(MAP_DIR / "address_id_map_labels.parquet",
                                                        compression="zstd", statistics=True)
    print(f"[✓] labels: {targets_global.height:,} узлов")

# -------------------- 2) edges_all --------------------


def build_edges_all():
    if not any(SRC_TX_DIR.glob("*.parquet")):
        raise FileNotFoundError(
            f"Не найдены входные транзакции в {SRC_TX_DIR}")
    lf_edges_all = (
        pl.scan_parquet(str(SRC_TX_DIR / "*.parquet"))
        .select(
            pl.col("from_address").str.to_lowercase().alias("from_lc"),
            pl.col("to_address").str.to_lowercase().alias("to_lc"),
            pl.col("timestamp"),
            pl.col("value_wei").cast(pl.Utf8).alias(
                "value_wei"),  # строки → точность и совместимость
            pl.col("tx_fee_wei").cast(pl.Utf8).alias("tx_fee_wei"),
            pl.col("block_number"),
            pl.col("contract_creation"),
            pl.col("tx_hash"),
        )
        .with_columns([
            pl.col("from_lc").hash(
                **HASH_SEEDS).cast(pl.UInt64).alias("src_id"),
            pl.col("to_lc").hash(**HASH_SEEDS).cast(pl.UInt64).alias("dst_id"),
            pl.col("timestamp").dt.epoch("s").cast(pl.Int64).alias("ts"),
            pl.col("block_number").cast(pl.Int64),
            pl.col("contract_creation").cast(pl.Boolean),
        ])
        .select(["src_id", "dst_id", "ts", "value_wei", "tx_fee_wei", "block_number", "contract_creation", "tx_hash"])
    )
    lf_edges_all.sink_parquet(
        str(EDGES_ALL_FP), compression="zstd", statistics=True)
    print(f"[✓] edges_all → {EDGES_ALL_FP}")

# -------------------- 3) маленький индекс окон (month/week) --------------------


def build_window_index():
    lf = (
        pl.scan_parquet(str(EDGES_ALL_FP))
        .select(
            pl.col("ts"),
            pl.from_epoch("ts", time_unit="s").alias("ts_dt"),
        )
        .with_columns([
            pl.col("ts_dt").dt.strftime("%Y-%m").alias("month"),
            pl.col("ts_dt").dt.strftime("%G-W%V").alias("week"),
        ])
    )
    month_idx = collect_streaming(
        lf.group_by("month").agg([
            pl.col("ts").min().alias("window_min_ts"),
            pl.col("ts").max().alias("window_end_ts"),
            pl.len().alias("edges_cnt"),
        ]).sort("month")
    )
    week_idx = collect_streaming(
        lf.group_by("week").agg([
            pl.col("ts").min().alias("window_min_ts"),
            pl.col("ts").max().alias("window_end_ts"),
            pl.len().alias("edges_cnt"),
        ]).sort("week")
    )
    # Сохраняем — nodes_cnt добавим итеративно (дёшево, нет большой unique())
    month_idx.write_parquet(
        META_DIR / "month_window_meta.parquet", compression="zstd", statistics=True)
    week_idx.write_parquet(
        META_DIR / "week_window_meta.parquet", compression="zstd", statistics=True)
    print(f"[✓] window index: {len(month_idx)} months, {len(week_idx)} weeks")

# -------------------- 4) итеративно считаем nodes_cnt и targets per window --------------------


def enrich_meta_and_build_targets():
    tg_ids = pl.read_parquet(
        LABELS_DIR / "targets_global.parquet").select("node_id").unique()
    tg_ids = tg_ids.with_columns(pl.col("node_id").cast(pl.UInt64))  # тип явно

    # helper: обработка любого freq
    def process_freq(freq: str, meta_fp: Path, targets_out_fp: Path):
        meta = pl.read_parquet(meta_fp).sort(freq)
        nodes_cnt_list = []
        targets_chunks = []
        chunk_size = 50  # каждые 50 окон будем флэшить на диск

        for i, row in enumerate(meta.iter_rows(named=True)):
            win_id = row[freq]
            ts_min = int(row["window_min_ts"])
            ts_max = int(row["window_end_ts"])

            # Рёбра окна (только нужный ts-диапазон)
            lf_win = (
                pl.scan_parquet(str(EDGES_ALL_FP))
                .filter((pl.col("ts") >= ts_min) & (pl.col("ts") <= ts_max))
                .select([pl.col("src_id").cast(pl.UInt64), pl.col("dst_id").cast(pl.UInt64)])
            )

            # Уникальные узлы окна (только внутри окна → дёшево)
            nodes_win = collect_streaming(
                pl.concat([
                    lf_win.select(pl.col("src_id").alias("node_id")),
                    lf_win.select(pl.col("dst_id").alias("node_id")),
                ], how="vertical").unique()
            )
            nodes_cnt_list.append((win_id, len(nodes_win)))

            # targets окна = пересечение с tg_ids (114k) — дешёво
            targets_win = nodes_win.join(tg_ids, on="node_id", how="inner")
            if targets_win.height > 0:
                targets_win = targets_win.with_columns(
                    pl.lit(win_id).alias(freq)).select([freq, "node_id"])
                targets_chunks.append(targets_win)

            # периодически флашим targets, чтобы не держать много в RAM
            if len(targets_chunks) >= chunk_size:
                df = pl.concat(targets_chunks, how="vertical")
                mode = "wb" if not targets_out_fp.exists() else "ab"
                # Политика: пишем одним файлом — проще для downstream
                if not targets_out_fp.exists():
                    df.write_parquet(
                        targets_out_fp, compression="zstd", statistics=True)
                else:
                    # Append: polars напрямую не аппендит в parquet.
                    # Решение: читаем старое быстро (это нечасто) и дописываем объединённое.
                    old = pl.read_parquet(targets_out_fp)
                    pl.concat([old, df], how="vertical").write_parquet(
                        targets_out_fp, compression="zstd", statistics=True)
                targets_chunks = []

            if (i + 1) % 25 == 0:
                print(f"[{freq}] processed {i+1}/{meta.height} windows")

        # финальный флаш targets
        if targets_chunks:
            df = pl.concat(targets_chunks, how="vertical")
            if not targets_out_fp.exists():
                df.write_parquet(
                    targets_out_fp, compression="zstd", statistics=True)
            else:
                old = pl.read_parquet(targets_out_fp)
                pl.concat([old, df], how="vertical").write_parquet(
                    targets_out_fp, compression="zstd", statistics=True)

        # записываем nodes_cnt обратно в meta (не держим гигантов)
        if nodes_cnt_list:
            add = pl.DataFrame({freq: [a for a, _ in nodes_cnt_list], "nodes_cnt": [
                               b for _, b in nodes_cnt_list]})
            meta_new = meta.join(add, on=freq, how="left")
            meta_new.write_parquet(
                meta_fp, compression="zstd", statistics=True)
            print(f"[✓] {freq}: nodes_cnt filled; targets → {targets_out_fp}")

    process_freq("month", META_DIR / "month_window_meta.parquet",
                 TARGETS_DIR / "month_targets.parquet")
    process_freq("week",  META_DIR / "week_window_meta.parquet",
                 TARGETS_DIR / "week_targets.parquet")

# -------------------- 5) экспорт помесячных/понедельных (низкая память) --------------------


def export_windows():
    month_meta = pl.read_parquet(
        META_DIR / "month_window_meta.parquet").sort("month")
    for month, ts_min, ts_max in zip(month_meta["month"], month_meta["window_min_ts"], month_meta["window_end_ts"]):
        out_dir = MONTH_DIR / f"month={month}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / "edges.parquet"
        if out_fp.exists():
            continue
        lf = (
            pl.scan_parquet(str(EDGES_ALL_FP))
              .filter((pl.col("ts") >= int(ts_min)) & (pl.col("ts") <= int(ts_max)))
              .select(["src_id", "dst_id", "ts", "value_wei", "tx_fee_wei", "block_number", "contract_creation", "tx_hash"])
        )
        df = collect_streaming(lf)
        df.write_parquet(out_fp, compression="zstd",
                         statistics=True, row_group_size=ROW_GROUP)

    week_meta = pl.read_parquet(
        META_DIR / "week_window_meta.parquet").sort("week")
    for week, ts_min, ts_max in zip(week_meta["week"], week_meta["window_min_ts"], week_meta["window_end_ts"]):
        out_dir = WEEK_DIR / f"week={week}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / "edges.parquet"
        if out_fp.exists():
            continue
        lf = (
            pl.scan_parquet(str(EDGES_ALL_FP))
              .filter((pl.col("ts") >= int(ts_min)) & (pl.col("ts") <= int(ts_max)))
              .select(["src_id", "dst_id", "ts", "value_wei", "tx_fee_wei", "block_number", "contract_creation", "tx_hash"])
        )
        df = collect_streaming(lf)
        df.write_parquet(out_fp, compression="zstd",
                         statistics=True, row_group_size=ROW_GROUP)

    print("[✓] edges_by_month & edges_by_week готовы")

# -------------------- 6) README --------------------


def write_readme():
    README_FP.write_text(f"""# Fraud LSTM+GNN — Unified Dataset (gnn_dataset)

## Состав
- `edges_all/edges.parquet` — **все** транзакции одним файлом.
- `edges_by_month/month=YYYY-MM/edges.parquet` — транзакции за месяц.
- `edges_by_week/week=YYYY-Www/edges.parquet` — транзакции за ISO‑неделю.
- `meta/month_window_meta.parquet`, `meta/week_window_meta.parquet` — интервалы `ts` и статистики.
- `labels/targets_global.parquet` — метки адресов: `(node_id, is_scam, is_contract, address)`.
- `mapping/address_id_map_labels.parquet` — `(address, node_id)` для адресов из labels.
- `targets/month_targets.parquet`, `targets/week_targets.parquet` — `(окно, node_id)` для обучения.

## Схема edges
- `src_id`: UInt64 — детерминированный хэш от `lower(address)` (seed=20250823/17/31/73).
- `dst_id`: UInt64
- `ts`: Int64 — Unix‑время (секунды).
- `value_wei`: Utf8 — точное значение wei (строка).
- `tx_fee_wei`: Utf8 — точное значение комиссии в wei (строка).
- `block_number`: Int64
- `contract_creation`: Boolean
- `tx_hash`: Utf8

## Примечания
- Транзакции **не фильтровались**: в `edges_*` входят все рёбра.
- Лосс в обучении считаем **только** по адресам из `labels/targets_global.parquet`.
- Для динамики используйте `edges_by_month/*` или `edges_by_week/*`,
  либо фильтруйте `edges_all/edges.parquet` по `ts` (predicate‑pushdown поддерживается).
""", encoding="utf-8")
    print(f"[✓] README → {README_FP}")


# -------------------- main --------------------
if __name__ == "__main__":
    build_labels_and_mapping()
    build_edges_all()
    build_window_index()                 # маленький список окон (ts_min/max)
    enrich_meta_and_build_targets()      # итеративно: nodes_cnt + targets per window
    export_windows()                     # помесячно/понедельно → по одному окну
    write_readme()
    print(f"\n[✓] Единый датасет готов → {DS_ROOT}")
