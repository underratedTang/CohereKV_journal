#!/usr/bin/env python3
"""Shared K-only lambda sweeps: identical thresholds for both MIS priorities."""
from __future__ import annotations
import math


def sweep_settings(ee_lambdas):
    values = []
    for part in str(ee_lambdas).split(','):
        if not part.strip():
            continue
        value = float(part)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('MIS lambda must be finite and positive')
        if value in values:
            raise ValueError('Duplicate MIS lambda')
        values.append(value)
    if not values:
        raise ValueError('Empty MIS lambda sweep')
    return [dict(sweep_id=i + 1, ee_lambda=lam) for i, lam in enumerate(values)]


def sweep_label(s):
    return f'point={s["sweep_id"]} K-only-lambda={s["ee_lambda"]:g}'
