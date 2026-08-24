import pytest

from meridian.control_selection import ORIGINAL_SUITES, select_original_distribution_episodes


def _catalog() -> list[dict]:
    entries = []
    counts = {
        "libero_spatial": 10,
        "libero_object": 10,
        "libero_goal": 10,
        "libero_10": 10,
        "libero_90": 90,
    }
    for suite, count in counts.items():
        for index in range(count):
            entries.append(
                {
                    "type": "file",
                    "path": f"{suite}/task_{index}_demo.hdf5",
                    "size": 100 + index,
                    "lfs": {"oid": f"{index + 1:064x}"},
                }
            )
    return entries


def test_original_selection_is_nested_balanced_and_reproducible() -> None:
    selected = select_original_distribution_episodes(
        _catalog(), seed=73180, maximum_dose=8, dataset_root="/datasets"
    )
    assert len(selected) == 8
    suites = [item["suite"] for item in selected]
    assert suites[:4] == list(ORIGINAL_SUITES)
    assert suites[4:] == list(ORIGINAL_SUITES)
    assert select_original_distribution_episodes(
        _catalog(), seed=73180, maximum_dose=8, dataset_root="/datasets"
    ) == selected


def test_original_selection_rejects_incomplete_catalog() -> None:
    with pytest.raises(ValueError, match="unexpected LIBERO task catalog"):
        select_original_distribution_episodes(
            _catalog()[:-1], seed=1, maximum_dose=8, dataset_root="/datasets"
        )
