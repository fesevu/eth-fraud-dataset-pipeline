#!/usr/bin/env python3
"""
Объединение смешанных файлов .parquet и .parquet.gz в набор .parquet-частей.
Особенности:
- Автодетект формата: gzip vs parquet по сигнатурам (без доверия к расширению).
- Если уже parquet — читаем файл напрямую (без копирования/удаления).
- Если gzip — распаковываем во временный .parquet, читаем и удаляем tmp.
- Деление итогов по числу строк (-- MAX_ROWS_PER_FILE), запись row-group'ами.
"""

import sys
import os
import tempfile
import subprocess
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# ====== НАСТРОЙКИ ПОД ТВОЙ КЕЙС ======
IN_DIR = Path("/mnt/d/new_Fraud/dataset/final/GNN/row_gz")
OUT_DIR = Path("/mnt/d/new_Fraud/dataset/final/GNN/parquet")
# можно оставить так; скрипт сам поймёт формат
PATTERN = "transactions_daily-*.parquet.gz"

OUT_BASE = "transactions_daily"
CODEC = "zstd"                  # "zstd" | "snappy" | "gzip" | "brotli" | "lz4_raw"
MAX_ROWS_PER_FILE = 40_000_000   # 0 = всё в один файл; иначе делим по N строк
ROW_GROUP_SIZE = 1_000_000
# быстрее распаковывать .gz, если pigz установлен (sudo apt install pigz)
USE_PIGZ = True
UNIFY_SCHEMAS = False           # True, если у частей могут отличаться схемы
# =====================================

GZIP_MAGIC = b"\x1f\x8b"
PARQUET_MAGIC = b"PAR1"


def head(path: Path, n: int) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except Exception:
        return b""


def tail(path: Path, n: int) -> bytes:
    try:
        sz = path.stat().st_size
        if sz < n:
            return b""
        with open(path, "rb") as f:
            f.seek(sz - n)
            return f.read(n)
    except Exception:
        return b""


def is_gzip_file(path: Path) -> bool:
    return head(path, 2) == GZIP_MAGIC


def is_parquet_file(path: Path) -> bool:
    return head(path, 4) == PARQUET_MAGIC and tail(path, 4) == PARQUET_MAGIC


def get_parquet_handle(src: Path, tmpdir: Path):
    """
    Возвращает кортеж (path_to_parquet, is_temp).
    - Если src.gz: распакует во временный .parquet и вернёт (tmp, True).
    - Если src уже parquet: вернёт (src, False).
    - Если неизвестный формат: бросит исключение.
    """
    if src.stat().st_size == 0:
        raise RuntimeError(f"{src.name}: пустой файл")

    if is_gzip_file(src):
        # распаковываем .gz -> tmp/*.parquet
        dst = tmpdir / src.stem  # убираем .gz
        cmd = ["pigz" if USE_PIGZ else "gzip", "-dc", str(src)]
        with open(dst, "wb") as out:
            subprocess.run(cmd, stdout=out, check=True)
        if dst.stat().st_size == 0:
            raise RuntimeError(
                f"{src.name}: после распаковки размер 0 (битый gzip?)")
        return dst, True

    if is_parquet_file(src):
        # уже parquet — читаем напрямую, не трогаем оригинал
        return src, False

    sig = head(src, 16)
    raise RuntimeError(
        f"{src.name}: неизвестный формат (не gzip и не parquet). First bytes: {sig!r}")


def collect_unified_schema(files, tmpdir):
    schemas = []
    for p in files:
        loc, is_temp = get_parquet_handle(p, tmpdir)
        try:
            pf = pq.ParquetFile(loc)
            schemas.append(pf.schema_arrow)
        finally:
            if is_temp:
                Path(loc).unlink(missing_ok=True)
    return pa.unify_schemas(schemas)


def open_new_writer(idx: int, schema: pa.Schema):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{OUT_BASE}_part-{idx:05d}.parquet"
    writer = pq.ParquetWriter(out_path, schema=schema,
                              compression=CODEC, write_statistics=True)
    return writer, out_path


def main():
    files = sorted(IN_DIR.glob(PATTERN))
    if not files:
        print("Нет входных файлов по PATTERN.", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)

        # 1) Схема
        if UNIFY_SCHEMAS:
            print("[i] Unifying schemas...", file=sys.stderr)
            schema = collect_unified_schema(files, tmpdir)
        else:
            loc0, is_temp0 = get_parquet_handle(files[0], tmpdir)
            try:
                schema = pq.ParquetFile(loc0).schema_arrow
            finally:
                if is_temp0:
                    Path(loc0).unlink(missing_ok=True)

        # 2) Запись
        idx = 0
        writer, current = open_new_writer(idx, schema)
        rows_in_file = 0
        total_rows = 0

        try:
            for p in files:
                try:
                    loc, is_temp = get_parquet_handle(p, tmpdir)
                except Exception as e:
                    print(f"[!] Пропускаю {p.name}: {e}", file=sys.stderr)
                    continue

                try:
                    pf = pq.ParquetFile(loc)
                    if not UNIFY_SCHEMAS and pf.schema_arrow != schema:
                        raise RuntimeError(
                            f"Schema mismatch в {p.name}. Поставь UNIFY_SCHEMAS=True.")

                    for rg in range(pf.num_row_groups):
                        tbl = pf.read_row_group(rg)
                        if UNIFY_SCHEMAS and tbl.schema != schema:
                            tbl = tbl.cast(schema, safe=False)

                        # Деление по MAX_ROWS_PER_FILE
                        if MAX_ROWS_PER_FILE and rows_in_file + tbl.num_rows > MAX_ROWS_PER_FILE:
                            need = MAX_ROWS_PER_FILE - rows_in_file
                            if need > 0:
                                writer.write_table(
                                    tbl.slice(0, need), row_group_size=ROW_GROUP_SIZE)
                                total_rows += need
                            writer.close()
                            print(f"[i] wrote {current.name}", file=sys.stderr)

                            idx += 1
                            writer, current = open_new_writer(idx, schema)
                            rows_in_file = 0

                            rest = tbl.slice(need)
                            if rest.num_rows > 0:
                                writer.write_table(
                                    rest, row_group_size=ROW_GROUP_SIZE)
                                rows_in_file += rest.num_rows
                                total_rows += rest.num_rows
                        else:
                            writer.write_table(
                                tbl, row_group_size=ROW_GROUP_SIZE)
                            rows_in_file += tbl.num_rows
                            total_rows += tbl.num_rows
                finally:
                    if is_temp:
                        Path(loc).unlink(missing_ok=True)
        finally:
            writer.close()
            print(f"[i] wrote {current.name}", file=sys.stderr)

    print(f"[✓] Done. Out={OUT_DIR}, files={idx+1}, total_rows={total_rows}")


if __name__ == "__main__":
    main()
