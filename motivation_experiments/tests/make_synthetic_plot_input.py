#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


out = Path(__file__).resolve().parent / "_synthetic"
(out / "raw").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)

budgets = [0.2, 0.4, 0.6, 0.8]
taus = [0.7, 0.8, 0.9, 0.95]
record_tau = 0.8

# Synthetic pooled prevalence with the same qualitative behavior expected from
# the real experiment: tighter budget yields more unsafe many-to-one groups.
rate_table = {
    0.7: [0.44, 0.25, 0.16, 0.11],
    0.8: [0.52, 0.37, 0.29, 0.24],
    0.9: [0.49, 0.39, 0.33, 0.29],
    0.95: [0.25, 0.19, 0.17, 0.16],
}
pair_rate_table = {
    0.7: [0.12, 0.13, 0.13, 0.14],
    0.8: [0.20, 0.21, 0.21, 0.21],
    0.9: [0.27, 0.28, 0.29, 0.30],
    0.95: [0.31, 0.32, 0.33, 0.34],
}
rows = []
for tau in taus:
    for rho, rate in zip(budgets, rate_table[tau]):
        eligible = 2000
        unsafe = int(round(rate * eligible))
        pair_rate = pair_rate_table[tau][budgets.index(rho)]
        rows.append(
            {
                "model": "synthetic/model",
                "dataset": "synthetic",
                "anchor_budget": rho,
                "cosine_threshold": tau,
                "num_observation_units": 12,
                "eligible_groups": eligible,
                "unsafe_groups": unsafe,
                "unsafe_group_rate": unsafe / eligible,
                "candidate_pairs": 5000,
                "conflict_pairs": int(round(pair_rate * 5000)),
                "conflict_pair_rate": int(round(pair_rate * 5000)) / 5000,
                "assigned_candidates": 4000,
                "candidate_acceptance_rate": 0.7,
            }
        )

summary_path = out / "summary_overall.csv"
with summary_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

all_budget = []
all_min_leaf = []
all_unsafe = []
all_z = []
all_conflict_fraction = []
for idx, rho in enumerate(budgets):
    n = 2000
    unsafe_count = int(round(rate_table[record_tau][idx] * n))
    unsafe = np.zeros(n, dtype=bool)
    unsafe[rng.choice(n, size=unsafe_count, replace=False)] = True
    # Unsafe groups have a broad worst-pair cosine distribution below tau;
    # safe groups sit just above tau. This only validates the plotting path.
    unsafe_values = np.clip(rng.normal(0.70 + 0.012 * idx, 0.055, size=n), 0.42, record_tau - 0.002)
    safe_values = np.clip(rng.normal(0.86, 0.025, size=n), record_tau + 0.002, 0.98)
    min_leaf = np.where(unsafe, unsafe_values, safe_values)
    theta = np.arccos(np.clip(min_leaf, -1.0, 1.0))
    z = theta / np.arccos(record_tau)

    # Structural severity: fraction of leaf-leaf pairs below the same tau.
    # Unsafe groups get a broad positive density; safe groups have density 0.
    unsafe_density = np.clip(rng.beta(1.7, 2.5, size=n), 0.02, 1.0)
    conflict_fraction = np.where(unsafe, unsafe_density, 0.0)

    all_budget.append(np.full(n, rho, dtype=np.float64))
    all_min_leaf.append(min_leaf)
    all_unsafe.append(unsafe.astype(np.int64))
    all_z.append(z)
    all_conflict_fraction.append(conflict_fraction)

np.savez_compressed(
    out / "raw" / "group_records_plot_tau.npz",
    min_leaf_cosine=np.concatenate(all_min_leaf),
    anchor_budget=np.concatenate(all_budget),
    unsafe=np.concatenate(all_unsafe),
    normalized_max_angle=np.concatenate(all_z),
    conflict_pair_fraction=np.concatenate(all_conflict_fraction),
    record_tau=np.asarray([record_tau], dtype=np.float64),
    anchor_budgets=np.asarray(budgets, dtype=np.float64),
)

(out / "run_config.json").write_text(
    json.dumps(
        {
            "model": "synthetic/model",
            "dataset": "synthetic",
            "anchor_budgets": budgets,
            "cosine_thresholds": taus,
            "record_tau": record_tau,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(out)
