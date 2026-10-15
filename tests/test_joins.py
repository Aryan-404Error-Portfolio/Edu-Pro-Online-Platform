import pandas as pd

from src.data_loader import load_raw_tables, normalize_types
from src.cleaning import build_master


def test_joins_no_row_multiplication():
    raw = load_raw_tables()
    norm = normalize_types(raw)
    users = __import__("src.cleaning", fromlist=["add_age_band"]).add_age_band(norm["Users"])
    master = build_master(norm["Transactions"], users, norm["Courses"], norm["Teachers"])
    assert len(master) == len(norm["Transactions"])
    assert master["UserID"].notna().all()
    # unmatched flags
    assert "_user_matched" in master.columns
    assert "_course_matched" in master.columns
    assert "_teacher_matched" in master.columns
    assert master["_user_matched"].all()
    assert master["_course_matched"].all()
    assert master["_teacher_matched"].all()


def test_join_keys_unique():
    raw = load_raw_tables()
    assert raw["Users"]["UserID"].is_unique
    assert raw["Courses"]["CourseID"].is_unique
    assert raw["Teachers"]["TeacherID"].is_unique
