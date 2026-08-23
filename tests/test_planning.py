import pytest

from meridian.planning import spread_init_state_index


def test_regression_state_spread_wraps_within_libero_states() -> None:
    assert [spread_init_state_index(repeat) for repeat in range(5)] == [0, 17, 34, 1, 18]


def test_regression_state_spread_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        spread_init_state_index(-1)
    with pytest.raises(ValueError, match="positive"):
        spread_init_state_index(0, state_count=0)
