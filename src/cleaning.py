"""Cleaning, age bands, referential integrity, and data-quality reporting."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

AGE_BANDS = ["Under 18", "18-25", "26-35", "36-45", "46 and above", "Unknown / missing / invalid age"]
VALID_AGE_MIN = 0
VALID_AGE_MAX = 100


def age_band(age) -> str:
    """Map an age to a non-overlapping PRD age band."""
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "Unknown / missing / invalid age"
    if pd.isna(a):
        return "Unknown / missing / invalid age"
    if a < VALID_AGE_MIN or a > VALID_AGE_MAX:
        return "Unknown / missing / invalid age"
    if a < 18:
        return "Under 18"
    if a <= 25:
        return "18-25"
    if a <= 35:
        return "26-35"
    if a <= 45:
        return "36-45"
    return "46 and above"


def add_age_band(users: pd.DataFrame) -> pd.DataFrame:
    out = users.copy()
    out["AgeBand"] = pd.Categorical(
        out["Age"].apply(age_band), categories=AGE_BANDS, ordered=True
    )
    return out


def key_uniqueness_report(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    checks = [("Users", "UserID"), ("Courses", "CourseID"), ("Teachers", "TeacherID"), ("Transactions", "TransactionID")]
    for sheet, col in checks:
        df = tables[sheet]
        rows.append({
            "table": sheet,
            "key_column": col,
            "row_count": len(df),
            "unique_keys": df[col].nunique(dropna=False),
            "duplicate_keys": int(df[col].duplicated().sum()),
            "missing_keys": int(df[col].isna().sum()) if df[col].isna().any() else 0,
        })
    return pd.DataFrame(rows)


def missing_values_report(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sheet, df in tables.items():
        for col in df.columns:
            rows.append({
                "table": sheet,
                "column": col,
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": round(df[col].isna().mean() * 100, 2),
            })
    return pd.DataFrame(rows)


def duplicate_rows_report(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sheet, df in tables.items():
        rows.append({
            "table": sheet,
            "duplicate_full_rows": int(df.duplicated().sum()),
        })
    return pd.DataFrame(rows)


def invalid_age_report(users: pd.DataFrame, teachers: pd.DataFrame = None) -> pd.DataFrame:
    rows = []
    for name, df, col in [("Users", users, "Age"), ("Teachers", teachers, "Age")] if teachers is not None else [("Users", users, "Age")]:
        s = pd.to_numeric(df[col], errors="coerce")
        invalid = s.isna() | (s < VALID_AGE_MIN) | (s > VALID_AGE_MAX)
        rows.append({
            "table": name,
            "invalid_or_missing_age_count": int(invalid.sum()),
            "min_age": float(s.min()) if s.notna().any() else None,
            "max_age": float(s.max()) if s.notna().any() else None,
        })
    return pd.DataFrame(rows)


def invalid_range_report(courses: pd.DataFrame, teachers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(table, field, series, lo=None, hi=None):
        s = pd.to_numeric(series, errors="coerce")
        bad = s.isna()
        if lo is not None:
            bad = bad | (s < lo)
        if hi is not None:
            bad = bad | (s > hi)
        rows.append({"table": table, "field": field, "invalid_count": int(bad.sum()),
                     "min": float(s.min()) if s.notna().any() else None,
                     "max": float(s.max()) if s.notna().any() else None})

    add("Courses", "CoursePrice", courses["CoursePrice"], lo=0)
    add("Courses", "CourseDuration", courses["CourseDuration"], lo=0)
    add("Courses", "CourseRating", courses["CourseRating"], lo=0, hi=5)
    add("Teachers", "TeacherRating", teachers["TeacherRating"], lo=0, hi=5)
    add("Teachers", "YearsOfExperience", teachers["YearsOfExperience"], lo=0)
    add("Transactions", "Amount", transactions["Amount"], lo=0)
    # dates
    bad_dates = transactions["TransactionDate"].isna()
    rows.append({"table": "Transactions", "field": "TransactionDate", "invalid_count": int(bad_dates.sum()),
                 "min": str(transactions["TransactionDate"].min()) if transactions["TransactionDate"].notna().any() else None,
                 "max": str(transactions["TransactionDate"].max()) if transactions["TransactionDate"].notna().any() else None})
    return pd.DataFrame(rows)


def referential_integrity_report(transactions: pd.DataFrame, users: pd.DataFrame,
                                 courses: pd.DataFrame, teachers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({
        "check": "Transactions.UserID in Users",
        "total": len(transactions),
        "matched": int(transactions["UserID"].isin(users["UserID"]).sum()),
        "unmatched": int((~transactions["UserID"].isin(users["UserID"])).sum()),
    })
    rows.append({
        "check": "Transactions.CourseID in Courses",
        "total": len(transactions),
        "matched": int(transactions["CourseID"].isin(courses["CourseID"]).sum()),
        "unmatched": int((~transactions["CourseID"].isin(courses["CourseID"])).sum()),
    })
    rows.append({
        "check": "Transactions.TeacherID in Teachers",
        "total": len(transactions),
        "matched": int(transactions["TeacherID"].isin(teachers["TeacherID"]).sum()),
        "unmatched": int((~transactions["TeacherID"].isin(teachers["TeacherID"])).sum()),
    })
    # duplicate transaction rows
    rows.append({
        "check": "Duplicate full transaction rows",
        "total": len(transactions),
        "matched": int(transactions.duplicated().sum()),
        "unmatched": int((~transactions.duplicated()).sum()),
    })
    # duplicate user-course pairs
    dup_pairs = int(transactions.duplicated(subset=["UserID", "CourseID"]).sum())
    rows.append({
        "check": "Duplicate UserID+CourseID pairs (repeat enrollments)",
        "total": len(transactions),
        "matched": int(transactions[["UserID", "CourseID"]].drop_duplicates().shape[0]),
        "unmatched": dup_pairs,
    })
    return pd.DataFrame(rows)


def amount_price_discrepancy_report(transactions: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    m = transactions.merge(courses[["CourseID", "CoursePrice"]], on="CourseID", how="left", validate="many_to_one")
    diff = m["Amount"] != m["CoursePrice"]
    return pd.DataFrame([{
        "records_compared": len(m),
        "discrepant_records": int(diff.sum()),
        "discrepancy_pct": round(diff.mean() * 100, 2),
    }])


def build_master(transactions, users, courses, teachers) -> pd.DataFrame:
    """Left-join transactions to users/courses/teachers with unmatched flags. No row multiplication."""
    u = users.copy()
    c = courses.copy()
    te = teachers.copy()

    # avoid column name collisions for Age/Gender
    u = u.rename(columns={"Age": "Age_Learner", "Gender": "Gender_Learner"})
    te = te.rename(columns={"Age": "Age_Teacher", "Gender": "Gender_Teacher"})

    master = transactions.merge(u, on="UserID", how="left", validate="many_to_one", indicator=False)
    master["_user_matched"] = master["UserName"].notna() if "UserName" in master.columns else master["Age_Learner"].notna()
    # more robust matched flag
    master["_user_matched"] = master["UserID"].isin(users["UserID"])

    master = master.merge(c, on="CourseID", how="left", validate="many_to_one")
    master["_course_matched"] = master["CourseID"].isin(courses["CourseID"])

    master = master.merge(te, on="TeacherID", how="left", validate="many_to_one")
    master["_teacher_matched"] = master["TeacherID"].isin(teachers["TeacherID"])

    # Age band on learner age
    master["AgeBand"] = pd.Categorical(
        master["Age_Learner"].apply(age_band), categories=AGE_BANDS, ordered=True
    )

    # Normalise gender missing
    master["Gender_Learner"] = master["Gender_Learner"].fillna("Unknown / missing")
    master.loc[master["Gender_Learner"].isin(["nan", "None", "NaN", ""]), "Gender_Learner"] = "Unknown / missing"

    return master


def data_quality_summary(tables, master) -> dict:
    """High-level counts for notice."""
    issues = []
    if missing_values_report(tables)["missing_count"].sum() > 0:
        issues.append("Missing values present in source tables.")
    if key_uniqueness_report(tables)["duplicate_keys"].sum() > 0:
        issues.append("Duplicate primary keys present.")
    ri = referential_integrity_report(
        tables["Transactions"], tables["Users"], tables["Courses"], tables["Teachers"]
    )
    if (ri["unmatched"] > 0).any():
        issues.append("Unmatched foreign keys in transactions.")
    invalid_ages = invalid_age_report(tables["Users"], tables["Teachers"])
    if invalid_ages["invalid_or_missing_age_count"].sum() > 0:
        issues.append("Invalid or missing ages present.")
    if not issues:
        issues.append("No major data-quality issues detected in the supplied data.")
    return {"issues": issues}
