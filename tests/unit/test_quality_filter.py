from cpo.preference.quality_filter import filter_by_score


def test_quality_filter() -> None:
    rows = [{"chosen_score": 3, "rejected_score": 4}, {"chosen_score": 2, "rejected_score": 4}]
    out = filter_by_score(rows, min_score=3)
    assert len(out) == 1
