"""KPI and aggregation logic for EduPro descriptive analytics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .cleaning import AGE_BANDS


def total_enrollment_records(df: pd.DataFrame) -> int:
    return int(len(df))


def unique_learner_course_pairs(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df[["UserID", "CourseID"]].drop_duplicates().shape[0])


def distinct_courses_with_enrollments(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df["CourseID"].nunique())


def learners_with_enrollments(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df["UserID"].nunique())


def enrollments_by_age_group(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=["AgeBand", "enrollments", "pct"])
    counts = master["AgeBand"].value_counts().reindex(AGE_BANDS).fillna(0).astype(int)
    total = counts.sum()
    out = pd.DataFrame({"AgeBand": counts.index, "enrollments": counts.values})
    out["pct"] = (out["enrollments"] / total * 100) if total else 0.0
    return out


def learner_age_distribution(users: pd.DataFrame) -> pd.DataFrame:
    if users.empty:
        return pd.DataFrame(columns=["AgeBand", "learners", "pct"])
    counts = users["AgeBand"].value_counts().reindex(AGE_BANDS).fillna(0).astype(int)
    total = counts.sum()
    out = pd.DataFrame({"AgeBand": counts.index, "learners": counts.values})
    out["pct"] = (out["learners"] / total * 100) if total else 0.0
    return out


def gender_participation_learners(users: pd.DataFrame) -> pd.DataFrame:
    if users.empty:
        return pd.DataFrame(columns=["Gender", "learners", "pct"])
    counts = users["Gender"].value_counts()
    total = counts.sum()
    out = pd.DataFrame({"Gender": counts.index, "learners": counts.values})
    out["pct"] = out["learners"] / total * 100 if total else 0.0
    return out


def gender_enrollment_counts(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=["Gender", "enrollments", "pct"])
    counts = master["Gender_Learner"].value_counts()
    total = counts.sum()
    out = pd.DataFrame({"Gender": counts.index, "enrollments": counts.values})
    out["pct"] = out["enrollments"] / total * 100 if total else 0.0
    return out


def category_popularity(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=["CourseCategory", "enrollments", "pct"])
    counts = master["CourseCategory"].value_counts().sort_values(ascending=False)
    total = counts.sum()
    out = pd.DataFrame({"CourseCategory": counts.index, "enrollments": counts.values})
    out["pct"] = out["enrollments"] / total * 100 if total else 0.0
    return out


def type_popularity(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=["CourseType", "enrollments", "pct"])
    counts = master["CourseType"].value_counts()
    total = counts.sum()
    out = pd.DataFrame({"CourseType": counts.index, "enrollments": counts.values})
    out["pct"] = out["enrollments"] / total * 100 if total else 0.0
    return out


def level_popularity(master: pd.DataFrame, level_order=None) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=["CourseLevel", "enrollments", "pct"])
    if level_order is None:
        level_order = ["Beginner", "Intermediate", "Advanced"]
    counts = master["CourseLevel"].value_counts().reindex(level_order).fillna(0).astype(int)
    # keep only levels present in reindex? reindex keeps all requested
    total = counts.sum()
    out = pd.DataFrame({"CourseLevel": counts.index, "enrollments": counts.values})
    out["pct"] = out["enrollments"] / total * 100 if total else 0.0
    return out


def courses_per_learner(master: pd.DataFrame) -> pd.Series:
    if master.empty:
        return pd.Series(dtype=float)
    return master.groupby("UserID")["CourseID"].nunique()


def avg_courses_per_learner(master: pd.DataFrame) -> float:
    s = courses_per_learner(master)
    if s.empty:
        return 0.0
    return float(s.mean())


def median_courses_per_learner(master: pd.DataFrame) -> float:
    s = courses_per_learner(master)
    if s.empty:
        return 0.0
    return float(s.median())


def enrollment_concentration(master: pd.DataFrame, top_pct: float = 0.10) -> dict:
    s = courses_per_learner(master) if not master.empty else pd.Series(dtype=float)
    # concentration by enrollment records, not distinct courses
    per_user_records = master.groupby("UserID").size() if not master.empty else pd.Series(dtype=int)
    if per_user_records.empty:
        return {"top_n": 0, "total_learners": 0, "top_share_pct": 0.0}
    sorted_counts = per_user_records.sort_values(ascending=False)
    top_n = max(1, int(np.ceil(len(sorted_counts) * top_pct)))
    top_share = sorted_counts.head(top_n).sum() / sorted_counts.sum() * 100
    return {
        "top_n": int(top_n),
        "total_learners": int(len(sorted_counts)),
        "top_share_pct": float(top_share),
    }


def age_gender_crosstab(users: pd.DataFrame) -> pd.DataFrame:
    if users.empty:
        return pd.DataFrame()
    return pd.crosstab(users["AgeBand"], users["Gender"])


def heatmap_age_category(master: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    ct = pd.crosstab(master["AgeBand"], master["CourseCategory"])
    if normalize:
        # within age group percentage
        row_totals = ct.sum(axis=1).replace(0, np.nan)
        ct = (ct.div(row_totals, axis=0) * 100).fillna(0)
    return ct


def gender_level_comparison(master: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    ct = pd.crosstab(master["Gender_Learner"], master["CourseLevel"])
    if normalize:
        row_totals = ct.sum(axis=1).replace(0, np.nan)
        ct = (ct.div(row_totals, axis=0) * 100).fillna(0)
    return ct


def age_level_comparison(master: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    ct = pd.crosstab(master["AgeBand"], master["CourseLevel"])
    if normalize:
        row_totals = ct.sum(axis=1).replace(0, np.nan)
        ct = (ct.div(row_totals, axis=0) * 100).fillna(0)
    return ct


def gender_category_comparison(master: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    ct = pd.crosstab(master["Gender_Learner"], master["CourseCategory"])
    if normalize:
        row_totals = ct.sum(axis=1).replace(0, np.nan)
        ct = (ct.div(row_totals, axis=0) * 100).fillna(0)
    return ct


def course_level_summary(master: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    agg = master.groupby("CourseID").agg(
        enrollments=("TransactionID", "count"),
        unique_learners=("UserID", "nunique"),
    ).reset_index()
    agg = agg.merge(courses, on="CourseID", how="left", validate="one_to_one")
    agg = agg.sort_values("enrollments", ascending=False).reset_index(drop=True)
    return agg


def monthly_trend(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=["Month", "enrollments"])
    m = master.copy()
    m["Month"] = m["TransactionDate"].dt.to_period("M").astype(str)
    trend = m.groupby("Month").size().reset_index(name="enrollments")
    return trend


def quarterly_trend(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=["Quarter", "enrollments"])
    m = master.copy()
    m["Quarter"] = m["TransactionDate"].dt.to_period("Q").astype(str)
    trend = m.groupby("Quarter").size().reset_index(name="enrollments")
    return trend
