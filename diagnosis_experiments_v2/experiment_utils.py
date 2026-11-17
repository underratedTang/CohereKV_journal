from __future__ import annotations
from typing import List


def parse_float_csv(text: str) -> List[float]:
    values = [float(x.strip()) for x in str(text).split(',') if x.strip()]
    if not values:
        raise ValueError('Expected at least one floating-point value.')
    return values


def resolve_selection(text: str, total: int) -> List[int]:
    text = str(text).strip().lower()
    if text == 'all':
        return list(range(total))
    if ':' in text and ',' not in text:
        parts = text.split(':')
        if len(parts) not in (2, 3):
            raise ValueError(f'Invalid slice selection: {text}')
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else total
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        values = list(range(start, stop, step))
    else:
        values = [int(x.strip()) for x in text.split(',') if x.strip()]
    if not values:
        raise ValueError(f'Selection {text!r} resolved to no indices.')
    bad = [i for i in values if i < 0 or i >= total]
    if bad:
        raise ValueError(f'Selection {text!r} contains out-of-range indices {bad}; total={total}.')
    return values
