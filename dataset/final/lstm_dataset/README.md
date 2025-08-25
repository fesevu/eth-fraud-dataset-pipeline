# 🕸 Ethereum Transaction Graph Dataset (GNN)

This dataset represents **Ethereum transactions as edges** between addresses.  
It is designed for Graph Neural Networks (GNN), both **static embeddings** and **temporal graph learning**.  
The dataset contains 2-hop graphs, i.e., it includes neighbors of neighbors for each address.

---

## 📑 Contents

- `edges_all/edges.parquet` — all transactions (full edge list).
- `edges_by_week/week=YYYY-Www/edges.parquet` — weekly slices.
- `edges_by_month/month=YYYY-MM/edges.parquet` — monthly slices.
- `meta/{week,month}_window_meta.parquet` — time window ranges and statistics.
- `labels/targets_global.parquet` — labeled addresses `(node_id, is_scam, is_contract, address)`.
- `mapping/address_id_map_labels.parquet` — `(address, node_id)` mapping.
- `targets/{week,month}_targets.parquet` — labeled nodes active in each window.

---

## 🔑 Edge Schema

| Field            | Type   | Units    | Description |
|------------------|--------|----------|-------------|
| src_id           | UInt64 | —        | Source node ID (hash of lowercase Ethereum address). |
| dst_id           | UInt64 | —        | Destination node ID (hash of lowercase Ethereum address). |
| ts               | Int64  | seconds  | Unix timestamp of the transaction (UTC). |
| value_wei        | STRING | wei      | Transaction value in wei (exact decimal stored as string). |
| tx_fee_wei       | STRING | wei      | Transaction fee in wei (exact decimal stored as string). |
| block_number     | Int64  | block    | Ethereum block number of the transaction. |
| contract_creation| Bool   | —        | True if transaction created a smart contract. |
| tx_hash          | STRING | hex      | Unique transaction hash. |

---

## Notes

- Transactions are **not filtered**: all edges included.  
- **Supervision**: loss computed only on labeled addresses.  
- **Dynamic GNN**: use `edges_by_week/` or `edges_by_month/`.  
- **Static embeddings**: use `edges_all/edges.parquet`.  
