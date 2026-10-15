import pandas as pd

from src.data_loader import load_raw_tables, normalize_types
from src.cleaning import add_age_band, build_master


def _master():
    raw = load_raw_tables()
    norm = normalize_types(raw)
    users = add_age_band(norm["Users"])
    return build_master(norm["Transactions"], users, norm["Courses"], norm["Teachers"])


def test_filter_reduces_rows():
    master = _master()
    sub = master[master["CourseLevel"] == "Beginner"]
    assert len(sub) < len(master)
    assert len(sub) > 0


def test_filter_empty_result_handled():
    master = _master()
    sub = master[master["CourseLevel"] == "NonexistentLevel"]
    assert len(sub) == 0
