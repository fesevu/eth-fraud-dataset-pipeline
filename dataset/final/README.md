# 🕸 Ethereum Address Behavior Dataset — GNN + LSTM (Fraud Detection)

This dataset is designed for **fraud detection on Ethereum addresses** using a **dual-modality approach**:
- **Graph Neural Networks (GNN):** transaction graph structure.
- **Recurrent Models (LSTM/Transformers):** time-series of address features.

The dataset is built from:
- **Ethereum public BigQuery dataset** (`bigquery-public-data.crypto_ethereum.transactions`).
- **Etherscan labels + custom scam labels**.
- **Balanced address list** of ~115k addresses (scam vs non-scam, contracts vs EOAs).

## 📦 Dataset Collection Pipeline

To reproduce or customize the dataset, use the instructions and code in the [eth-fraud-dataset-pipeline repository](https://github.com/fesevu/eth-fraud-dataset-pipeline).  
That repository provides:
- Scripts for downloading raw data from public sources (BigQuery, Etherscan, curated scam lists).
- Code for merging, deduplicating, and balancing address labels.
- Tools for building the GNN and LSTM datasets (parquet files, mappings, targets).
- Utilities for generating checksums and manifests for data integrity.

**You must run the provided scripts to generate the dataset locally; the data files are not stored in the GitHub repository.**
---

## 📂 Repository Structure

final/
├─ gnn_dataset/ # GNN dataset (edges, meta, labels, mapping, targets)
│ ├─ edges_all/edges.parquet
│ ├─ edges_by_week/week=YYYY-Www/edges.parquet
│ ├─ edges_by_month/month=YYYY-MM/edges.parquet
│ ├─ meta/{week,month}_window_meta.parquet
│ ├─ labels/targets_global.parquet
│ ├─ mapping/address_id_map_labels.parquet
│ ├─ targets/{week,month}_targets.parquet
│ └─ README.md
└─ lstm_dataset/ # LSTM dataset (daily → weekly → monthly aggregations)
├─ daily_filtered.parquet
├─ weekly.parquet
├─ monthly.parquet
└─ README.md

- `gnn_dataset/` → GNN dataset (graph edges, slices, labels, mapping).
- `lstm_dataset/` → LSTM dataset (tabular features, time-series).

---

## 🔑 Synchronization Between GNN and LSTM

- Both use the same **address universe** (`node_id` mapping).
- Both use the same **time windows**:
  - ISO weeks (`YYYY-Www`) from `gnn_dataset/meta/week_window_meta.parquet`.
  - Months (`YYYY-MM`) from `gnn_dataset/meta/month_window_meta.parquet`.

---

## 📂 Raw Data

Alongside the processed datasets, we also provide the **raw parquet exports** (all parquet files are compressed with **Zstandard (zstd)**):

final/
├─ GNN/parquet/ # raw transaction parquet chunks for GNN
│ ├─ transactions_daily_part-00000.parquet
│ ├─ transactions_daily_part-00001.parquet
│ └─ ...
├─ LSTM/parquet/ # raw daily features parquet for LSTM
│ ├─ daily_final_part-00000.parquet
│ ├─ daily_final_part-00001.parquet
│ └─ ...
├─ addr_labels_balanced.csv # balanced address list with labels
├─ addr_labels_balanced.csv # balanced subset with labels (used in GNN + LSTM)

---

### Contents
- **`GNN/parquet/`** — raw transaction-level parquet files, containing:
  - `from_address`, `to_address` (STRING, lowercase hex)  
  - `block_number` (INT64)  
  - `timestamp` (TIMESTAMP, UTC)  
  - `value_wei`, `tx_fee_wei` (NUMERIC in source, stored as string later)  
  - `nonce`, `input_data_size`, `contract_creation`, `tx_hash`, `day`
- **`LSTM/parquet/`** — raw daily activity parquet files (address-day features before filtering).
- **`addr_labels_big.csv`** — initial large list of Ethereum addresses (>1M), with scam/contract metadata, **not used directly** (later downsampled & balanced).
- **`addr_labels_balanced.csv`** — final balanced list of ~115k addresses (scam vs non-scam, contract vs EOA), used for both **GNN** and **LSTM** datasets.

All parquet files in this dataset are compressed using **Zstandard (zstd)** for efficient storage and fast access.

These files are the **starting point** for the preparation scripts:
- `build_unified_dataset.py` → creates `gnn_dataset/` (GNN).  
- `build_lstm_dataset_lowmem.py` → creates `lstm_dataset/` (LSTM).  

---

## ⚖️ Labels

- Source: Etherscan tags + curated scam lists.
- Balanced across:
  - **Scam vs Non-Scam**
  - **Contract vs EOA**
- Provided in:
  - `gnn_dataset/labels/targets_global.parquet`
  - `gnn_dataset/mapping/address_id_map_labels.parquet`

---

### Address Label Files

Both `addr_labels_big.csv` (full set) and `addr_labels_balanced.csv` (balanced subset) share the same schema:

| Field              | Type     | Units | Description |
|--------------------|----------|-------|-------------|
| address            | STRING   | hex   | Ethereum address (0x..., lowercase). |
| is_scam            | INT64    | 0/1   | Scam label: 1 = scam, 0 = non-scam. |
| description        | STRING   | —     | Free-text description (e.g. "Verified", "Phishing"). |
| activity_start_ts  | TIMESTAMP| UTC   | First observed activity timestamp. |
| activity_end_ts    | TIMESTAMP| UTC   | Last observed activity timestamp. |
| is_contract        | INT64    | 0/1   | Address type: 1 = smart contract, 0 = EOA. |

- **`addr_labels_big.csv`** — ~1M+ raw addresses with scam/contract metadata, **not used directly** (later downsampled and balanced).  
- **`addr_labels_balanced.csv`** — final balanced subset (~115k addresses, scam vs non-scam, contract vs EOA), used in both **GNN** and **LSTM** datasets.  

---

## 📦 Use Cases

- **Graph ML:** Train static embeddings (GraphSAGE, Node2Vec) or temporal GNNs.
- **Sequence ML:** Train LSTM/Transformer on address time-series features.
- **Fusion:** Combine GNN embeddings and LSTM features via `node_id`.
- **Fraud detection:** Predict scam addresses, contracts vs EOAs.

---

## 🛠 Collection Details

- Source: Ethereum mainnet via BigQuery.
- Labels: from Etherscan + custom curated lists.
- Timezone: UTC.
- ETH amounts stored as Decimal(38,9), exported as strings for precision.
- Data preparation optimized for BigQuery + Polars, fits in 12–24 GB RAM.

## 🔒 Integrity
- All files are checksummed (SHA256, optional MD5).
- See `CHECKSUMS.md` for a human-readable table.
- See `manifest.jsonl` for a machine-readable log (size, mtime, checksums).
- To verify after download:
  ```bash
  python3 make_checksums.py --verify --base /path/to/final

## 🗂 Source Datasets

The address list and labels (scam/non-scam, description) were compiled from the following public datasets:

- **Primary sources:**
  - [xblock.pro Dataset #13](https://xblock.pro/#/dataset/13)
  - [xblock.pro Dataset #25](https://xblock.pro/#/dataset/25)
  - [xblock.pro Dataset #50](https://xblock.pro/#/dataset/50)
  - [PTXPhish](https://github.com/blocksecteam/PTXPhish/tree/main?tab=readme-ov-file)
  - [Phishing Contract Sigmetrics](https://github.com/blocksecteam/phishing_contract_sigmetrics25/tree/main)
  - [Etherscan Open Source Contract Codes](https://etherscan.io/exportData?type=open-source-contract-codes)
  - [MyEtherWallet Ethereum Lists](https://github.com/MyEtherWallet/ethereum-lists)
  - [EtherScamDB](https://github.com/MrLuit/EtherScamDB/tree/master)
  - [CryptoScamDB Blacklist](https://github.com/CryptoScamDB/blacklist)
  - [ScamSniffer Scam Database](https://github.com/scamsniffer/scam-database)
  - [Forta Network Labelled Datasets](https://github.com/forta-network/labelled-datasets)
  - [Kaggle: Labelled Ethereum Addresses](https://www.kaggle.com/datasets/hamishhall/labelled-ethereum-addresses?select=eth_addresses.csv)
  - [Etherscan Labels](https://github.com/brianleect/etherscan-labels/tree/main/data/etherscan/combined)
  - [Kaggle: Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset/data)
  - [Ethereum Fraud Datasets](https://github.com/surajsjain/ethereum-fraud-datasets/tree/main)
  - [Kaggle: Ponzi Scheme Contracts](https://www.kaggle.com/datasets/polarwolf/ponzi-scheme-contracts-on-ethereum)
  - [Ethereum Fraud Detection](https://github.com/eltontay/Ethereum-Fraud-Detection)

- **Integration:**
  - Addresses and labels from these sources were merged and deduplicated.
  - The final balanced address list (~115k addresses) was constructed based on these datasets.

---
