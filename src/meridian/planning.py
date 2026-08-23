from __future__ import annotations


def spread_init_state_index(repeat: int, *, state_count: int = 50, stride: int = 17) -> int:
    if repeat < 0:
        raise ValueError("repeat must be non-negative")
    if state_count <= 0:
        raise ValueError("state_count must be positive")
    return (repeat * stride) % state_count
