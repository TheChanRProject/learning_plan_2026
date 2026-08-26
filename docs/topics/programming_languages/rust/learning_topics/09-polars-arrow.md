# Day 9 — Polars & Arrow

> **Goal:** Use Polars (Rust-first DataFrame) + Arrow/Parquet as the single-node / hot-path compute layer.

**Date:** Mon Sep 7, 2026

## What you'll learn

- Polars lazy engine + expression API
- Arrow columnar memory; Parquet as interchange
- Assemble stack: compute (Polars/DataFusion) + formats (Arrow/Parquet/Iceberg) + orchestrators
- Pattern: Polars in-node, scale out with Airflow/Dagster/K8s

## Prerequisites

[08 — DataFusion](./08-datafusion-spark-analog.md)

## Watch / read

- [Rust Learning Plan §2 Polars + §4 What's not there](../Rust_Learning_Plan.md)
- Polars Rust docs (getting started)

---

## Mental model

Polars ≈ **very fast single-machine Spark-like engine** (lazy plans, expressions). Distributed Polars exists but is early — not "drop on cluster like Spark." Pair with Arrow/Parquet for zero-copy-ish interchange across Python/Rust.

```text
Object storage (Parquet)
        ↓
   Polars lazy / DataFusion
        ↓
  Feature tables / model inputs
```

Typical assembly (from plan):

- DataFusion / Polars — compute
- Arrow / Parquet / Iceberg — storage & table layer
- Airflow / Dagster / K8s — orchestration

---

## Day 9 exercise

1. `cargo new day09_polars`
2. Add `polars` (features for CSV/Parquet as needed)
3. Build a `DataFrame` (or scan CSV), filter + group_by aggregate with expressions
4. Write same logical query note: "DataFusion SQL vs Polars expressions — which I'd use for X"

**Done when:** one aggregate prints; you can say when Polars beats DataFusion for your workflow (usually: exploratory / single-node ETL).

**Next:** [10 — Streaming & Agents](./10-streaming-agents-ml.md)
