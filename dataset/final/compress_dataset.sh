#!/usr/bin/env bash
set -euo pipefail

SRC="/mnt/d/new_Fraud/dataset/final"          # исходники (НЕ трогаем)
PUB="/mnt/d/new_Fraud/dataset/public_release" # копия для публикации
OUT="/mnt/d/new_Fraud/dataset"                # куда класть tar.zst
DRYRUN="${DRYRUN:-0}"
ZSTD_LEVEL="${ZSTD_LEVEL:-3}"
ROW_GROUP="${ROW_GROUP:-128000}"

echo "[i] Source (read-only): $SRC"
echo "[i] Public release:     $PUB"
echo "[i] Bundles out:        $OUT"

command -v python3 >/dev/null || { echo "[!] python3 not found"; exit 1; }
mkdir -p "$PUB" "$OUT"

# deps check
python3 - <<'PY'
import sys
try:
    import polars as pl
    import pyarrow.parquet as pq
except Exception:
    sys.stderr.write("[!] Need: pip install polars pyarrow zstandard\n"); raise
PY

# 0) copy READMEs/MD (без .py/.hidden, без parquet/csv/gz, и игнор row_gz/)
echo "[i] Copy READMEs & visible metadata..."
rsync -av \
  --include="*/" \
  --include="README.md" \
  --include="*.md" \
  --exclude="*.py" \
  --exclude=".*" \
  --exclude="*.parquet" \
  --exclude="*.csv" \
  --exclude="*.gz" \
  --exclude="row_gz/" \
  --exclude="_tmp_*" \
  "$SRC"/ "$PUB"/

# 1) Parquet -> public_release
# - если уже ZSTD: просто копируем 1:1
# - если не ZSTD: репакуем в ZSTD и валидируем row_count
echo "[i] Copy/Repack Parquet (skip already-ZSTD) with validation ..."
# find: обходим все *.parquet, игнорируем любые row_gz каталоги
while IFS= read -r -d '' src; do
  # prune row_gz
  case "$src" in *"/row_gz/"* ) continue ;; esac

  rel="${src#$SRC/}"
  dst="$PUB/$rel"
  dst_dir="$(dirname "$dst")"
  mkdir -p "$dst_dir"
  echo " → $rel"
  [ "$DRYRUN" = "1" ] && continue

  python3 - "$src" "$dst" "$ZSTD_LEVEL" "$ROW_GROUP" <<'PY'
import sys, os, shutil
import pyarrow.parquet as pq
import polars as pl

src, dst, lvl, rg = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

# Быстрая мета исходника
meta_src = pq.read_metadata(src)
rows_src = meta_src.num_rows

# Проверяем кодек первого столбца первой row-group (достаточно для решения)
# Если нет row_groups (теоретически), считаем как не ZSTD
is_zstd = False
if meta_src.num_row_groups > 0 and meta_src.row_group(0).num_columns > 0:
    try:
        codec = meta_src.row_group(0).column(0).compression
        # codec может быть 'ZSTD' или pyarrow enum -> приводим к строке
        is_zstd = str(codec).upper().endswith("ZSTD")
    except Exception:
        is_zstd = False

if is_zstd:
    # Просто копия 1:1 в public_release
    shutil.copy2(src, dst)
    # Доп. проверка row_count на копии (дёшево)
    rows_dst = pq.read_metadata(dst).num_rows
    if rows_src != rows_dst:
        try: os.remove(dst)
        except: pass
        raise SystemExit(f"[VALIDATION ERROR] (copy) Row-count mismatch: {rows_src} vs {rows_dst} for {dst}")
else:
    # Репакуем в ZSTD (стриминг)
    lf = pl.scan_parquet(src)
    lf.sink_parquet(dst, compression="zstd", statistics=True)
    # Валидация
    rows_dst = pq.read_metadata(dst).num_rows
    if rows_src != rows_dst:
        try: os.remove(dst)
        except: pass
        raise SystemExit(f"[VALIDATION ERROR] (repack) Row-count mismatch: {rows_src} vs {rows_dst} for {dst}")
PY
done < <(find "$SRC" -type d -name "row_gz" -prune -o -type f -name "*.parquet" -print0)

# 2) CSV -> .csv.zst (оставляя оригиналы). Игнор .gz и row_gz.
echo "[i] Compress CSV -> .csv.zst (skip *.gz, ignore row_gz/)..."
# берём только CSV в корне SRC (как у тебя labels); если есть глубже — можно сделать find
for csv in "$SRC"/*.csv; do
  [ -f "$csv" ] || continue
  base="$(basename "$csv")"
  out="$PUB/$base.zst"
  echo " → $base -> $(basename "$out")"
  [ "$DRYRUN" = "1" ] && continue
  zstd -19 -f -o "$out" "$csv"
done

# 3) Архивы из public_release (без .py/hidden и *.gz, игнор row_gz/)
echo "[i] Build tar.zst bundles from public_release ..."
if [ "$DRYRUN" != "1" ]; then
  tar -I "zstd -19 -T0" \
      --exclude='*.py' --exclude='.*' --exclude='*.gz' --exclude='row_gz' \
      -cf "$OUT/fraud_dataset_full.tar.zst" -C "$PUB" .

  [ -d "$PUB/dataset_v1" ] && tar -I "zstd -19 -T0" \
      --exclude='*.py' --exclude='.*' --exclude='*.gz' --exclude='row_gz' \
      -cf "$OUT/fraud_dataset_gnn.tar.zst" -C "$PUB" dataset_v1

  [ -d "$PUB/lstm_dataset" ] && tar -I "zstd -19 -T0" \
      --exclude='*.py' --exclude='.*' --exclude='*.gz' --exclude='row_gz' \
      -cf "$OUT/fraud_dataset_lstm.tar.zst" -C "$PUB" lstm_dataset

  # raw + labels (если есть)
  if [ -d "$PUB/GNN" ] || [ -d "$PUB/LSTM" ] || ls "$PUB"/*labels*.csv.zst >/dev/null 2>&1; then
    tar -I "zstd -19 -T0" \
        --exclude='*.py' --exclude='.*' --exclude='*.gz' --exclude='row_gz' \
        -cf "$OUT/fraud_dataset_raw.tar.zst" -C "$PUB" GNN LSTM ./*labels*.csv.zst 2>/dev/null || true
  fi
fi

# 4) Чексумы в public_release
echo "[i] Generate checksums in public_release ..."
if [ "$DRYRUN" != "1" ]; then
  python3 - <<'PY'
from subprocess import run
run(["python3","/mnt/d/new_Fraud/dataset/final/make_checksums.py","--base","/mnt/d/new_Fraud/dataset/public_release"], check=False)
PY
fi

echo "[✓] Done."
echo "    Originals intact at:  $SRC"
echo "    Public copy at:       $PUB"
echo "    Bundles at:           $OUT"
echo "    Checksums at:         $PUB/CHECKSUMS.md, $PUB/manifest.jsonl"
echo ""
echo "Tips:"
echo "  DRYRUN=1 bash compress_dataset_safe.sh     # preview only"
echo "  ZSTD_LEVEL=5 bash compress_dataset_safe.sh # stronger parquet compression"
