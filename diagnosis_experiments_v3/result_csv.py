#!/usr/bin/env python3
"""Atomic CSV output for heterogeneous per-method diagnostic records.

Individual methods are allowed to produce additional diagnostic fields.  Keep
columns in first-seen order and represent non-applicable fields as empty CSV
cells; never silently discard an extra field.
"""
from __future__ import annotations

import csv
from pathlib import Path


def save_csv(path, records):
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # DictWriter's default extrasaction='raise' rejects later H1-only fields
    # when the first row belongs to a physical-only control method.
    fields = list(dict.fromkeys(key for record in records for key in record))
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('w', encoding='utf8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, restval='')
            writer.writeheader()
            writer.writerows(records)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
