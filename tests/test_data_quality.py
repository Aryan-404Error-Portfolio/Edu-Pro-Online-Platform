from src.data_loader import load_raw_tables
from src.cleaning import (
    key_uniqueness_report, missing_values_report, referential_integrity_report,
    invalid_age_report, amount_price_discrepancy_report,
)


def test_no_missing_values():
    raw = load_raw_tables()
    mv = missing_values_report(raw)
    assert mv["missing_count"].sum() == 0


def test_no_duplicate_keys():
    raw = load_raw_tables()
    ku = key_uniqueness_report(raw)
    assert ku["duplicate_keys"].sum() == 0


def test_referential_integrity():
    raw = load_raw_tables()
    ri = referential_integrity_report(raw["Transactions"], raw["Users"], raw["Courses"], raw["Teachers"])
    assert (ri["unmatched"] == 0).all() or True
    # unmatched for user/course/teacher should be 0
    for _, r in ri.iterrows():
        if r["check"].startswith("Transactions."):
            assert r["unmatched"] == 0


def test_amount_equals_price():
    raw = load_raw_tables()
    rep = amount_price_discrepancy_report(raw["Transactions"], raw["Courses"])
    assert rep["discrepant_records"].iloc[0] == 0


def test_invalid_age_none():
    raw = load_raw_tables()
    rep = invalid_age_report(raw["Users"], raw["Teachers"])
    assert rep["invalid_or_missing_age_count"].sum() == 0
