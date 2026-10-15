import pandas as pd

from src.data_loader import load_raw_tables, normalize_types
from src.cleaning import add_age_band, build_master
from src.metrics import (
    total_enrollment_records, unique_learner_course_pairs, distinct_courses_with_enrollments,
    learners_with_enrollments, category_popularity, level_popularity, avg_courses_per_learner,
    median_courses_per_learner, enrollment_concentration, courses_per_learner,
)


def _master():
    raw = load_raw_tables()
    norm = normalize_types(raw)
    users = add_age_band(norm["Users"])
    return build_master(norm["Transactions"], users, norm["Courses"], norm["Teachers"]), users


def test_kpi_totals():
    master, users = _master()
    assert total_enrollment_records(master) == 10000
    assert unique_learner_course_pairs(master) == 10000
    assert distinct_courses_with_enrollments(master) == 60
    assert learners_with_enrollments(master) == 3000


def test_category_popularity_sums_to_100():
    master, _ = _master()
    cp = category_popularity(master)
    assert abs(cp["pct"].sum() - 100.0) < 1e-6
    assert (cp["enrollments"] > 0).all()


def test_level_popularity():
    master, _ = _master()
    lp = level_popularity(master)
    assert abs(lp["pct"].sum() - 100.0) < 1e-6


def test_avg_courses():
    master, _ = _master()
    assert abs(avg_courses_per_learner(master) - (10000 / 3000)) < 1e-6
    assert median_courses_per_learner(master) >= 1


def test_concentration():
    master, _ = _master()
    conc = enrollment_concentration(master)
    assert 0 < conc["top_share_pct"] <= 100
    assert conc["top_n"] >= 1


def test_empty_filter():
    master, _ = _master()
    empty = master.iloc[0:0]
    assert total_enrollment_records(empty) == 0
    assert unique_learner_course_pairs(empty) == 0
    assert distinct_courses_with_enrollments(empty) == 0
    assert learners_with_enrollments(empty) == 0
    assert avg_courses_per_learner(empty) == 0.0
    cp = category_popularity(empty)
    assert len(cp) == 0 or cp["enrollments"].sum() == 0
