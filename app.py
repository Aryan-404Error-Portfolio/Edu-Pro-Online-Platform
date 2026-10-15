"""EduPro Learner Demographics and Course Enrollment Behavior Dashboard.

Run with: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.data_loader import load_raw_tables, normalize_types, SHEET_NAMES, REQUIRED_COLUMNS
from src.cleaning import (
    AGE_BANDS, age_band, add_age_band, build_master,
    key_uniqueness_report, missing_values_report, duplicate_rows_report,
    invalid_age_report, invalid_range_report, referential_integrity_report,
    amount_price_discrepancy_report, data_quality_summary,
)
from src.metrics import (
    total_enrollment_records, unique_learner_course_pairs, distinct_courses_with_enrollments,
    learners_with_enrollments, enrollments_by_age_group, learner_age_distribution,
    gender_participation_learners, gender_enrollment_counts, category_popularity,
    type_popularity, level_popularity, courses_per_learner, avg_courses_per_learner,
    median_courses_per_learner, enrollment_concentration, age_gender_crosstab,
    heatmap_age_category, gender_level_comparison, age_level_comparison,
    gender_category_comparison, course_level_summary, monthly_trend,
)
from src.exports import to_csv_bytes, filter_selections_text

st.set_page_config(page_title="EduPro Learner Analytics", layout="wide", page_icon="📊")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading EduPro data...")
def load_all():
    raw_tables = load_raw_tables()
    norm_tables = normalize_types(raw_tables)
    users = add_age_band(norm_tables["Users"])
    courses = norm_tables["Courses"]
    transactions = norm_tables["Transactions"]
    teachers = norm_tables["Teachers"]
    # teachers age band not needed
    master = build_master(transactions, users, courses, teachers)
    # ensure AgeBand categorical on master already set
    return raw_tables, norm_tables, users, courses, transactions, teachers, master


try:
    raw_tables, norm_tables, users, courses, transactions, teachers, master = load_all()
except Exception as exc:
    st.error(f"Failed to load data: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.title("Global filters")

# Date range
min_date = master["TransactionDate"].min()
max_date = master["TransactionDate"].max()
if pd.isna(min_date):
    date_range = None
else:
    date_range = st.sidebar.date_input(
        "Transaction date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

age_options = AGE_BANDS
age_sel = st.sidebar.multiselect("Age group", age_options, default=age_options)

gender_options = sorted(master["Gender_Learner"].dropna().unique().tolist())
gender_sel = st.sidebar.multiselect("Recorded gender", gender_options, default=gender_options)

category_options = sorted(master["CourseCategory"].dropna().unique().tolist())
category_sel = st.sidebar.multiselect("Course category", category_options, default=category_options)

type_options = sorted(master["CourseType"].dropna().unique().tolist())
type_sel = st.sidebar.multiselect("Course type", type_options, default=type_options)

level_options = ["Beginner", "Intermediate", "Advanced"]
# restrict to those present
level_present = [l for l in level_options if l in set(master["CourseLevel"].dropna().unique())]
level_sel = st.sidebar.multiselect("Course level", level_present, default=level_present)

if st.sidebar.button("Reset filters"):
    st.rerun()

# Apply filters
filtered = master.copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2 and date_range[0] and date_range[1]:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    filtered = filtered[(filtered["TransactionDate"] >= start) & (filtered["TransactionDate"] < end)]

filtered = filtered[
    filtered["AgeBand"].astype(str).isin([str(x) for x in age_sel])
    & filtered["Gender_Learner"].isin(gender_sel)
    & filtered["CourseCategory"].isin(category_sel)
    & filtered["CourseType"].isin(type_sel)
    & filtered["CourseLevel"].isin(level_sel)
]

# Filtered users (age/gender only, for demographic denominators)
filtered_users = users.copy()
filtered_users = filtered_users[
    filtered_users["AgeBand"].astype(str).isin([str(x) for x in age_sel])
    & filtered_users["Gender"].isin(gender_sel)
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing {len(filtered):,} of {len(master):,} enrollment records")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("📊 EduPro — Learner Demographics & Enrollment Behavior")
st.caption("Descriptive analytics only. No prediction, monetization, or causal claims.")

# Data quality notice
dq = data_quality_summary(raw_tables, master)
if any("No major" not in i for i in dq["issues"]):
    with st.expander("⚠️ Data-quality notice", expanded=False):
        for i in dq["issues"]:
            st.write("- " + i)

# KPI cards
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registered learners", f"{len(filtered_users):,}", help="Registered learners matching age/gender filters.")
k2.metric("Learners with enrollments", f"{learners_with_enrollments(filtered):,}", help="Distinct learners in filtered enrollment records.")
k3.metric("Enrollment records", f"{total_enrollment_records(filtered):,}", help="Count of valid transaction rows after filters.")
k4.metric("Distinct courses", f"{distinct_courses_with_enrollments(filtered):,}", help="Distinct courses in filtered enrollment records.")
top_cat = filtered["CourseCategory"].value_counts().idxmax() if len(filtered) else "—"
k5.metric("Top category", str(top_cat))

st.markdown("---")

if len(filtered) == 0:
    st.warning("No data matches the selected filters. Please broaden your selection.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_demo, tab_enroll, tab_cross, tab_course, tab_quality = st.tabs(
    ["🏠 Overview", "👥 Learner Demographics", "📈 Enrollment Behavior",
     "🧭 Demographics × Preferences", "📚 Course Explorer", "🧪 Data Quality & Methodology"]
)

# ============================ Overview =====================================
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        ad = learner_age_distribution(filtered_users)
        fig = px.bar(ad, x="AgeBand", y="learners", title="Learner Age Distribution (registered learners)",
                     labels={"AgeBand": "Age band", "learners": "Learners"}, text="learners")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Denominator: registered learners matching age/gender filters.")
    with c2:
        gd = gender_participation_learners(filtered_users)
        fig = px.pie(gd, names="Gender", values="learners", title="Learner Gender Distribution", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Denominator: registered learners matching age/gender filters.")

    st.subheader("Enrollment volume over time")
    tr = monthly_trend(filtered)
    fig = px.line(tr, x="Month", y="enrollments", markers=True, title="Enrollment Records by Month")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Denominator: filtered enrollment records.")

    st.subheader("Quick KPIs")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Avg courses / learner", f"{avg_courses_per_learner(filtered):.2f}")
    cc2.metric("Median courses / learner", f"{median_courses_per_learner(filtered):.0f}")
    conc = enrollment_concentration(filtered)
    cc3.metric("Top 10% concentration", f"{conc['top_share_pct']:.1f}%")

# ============================ Demographics ==================================
with tab_demo:
    st.subheader("Age-band distribution")
    ad = learner_age_distribution(filtered_users)
    ad2 = ad.copy()
    ad2["pct"] = ad2["pct"].round(1)
    st.dataframe(ad2, use_container_width=True)
    fig = px.bar(ad, x="AgeBand", y="learners", title="Registered Learners by Age Band",
                 labels={"AgeBand": "Age band", "learners": "Learners"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recorded gender distribution")
    gd = gender_participation_learners(filtered_users)
    st.dataframe(gd, use_container_width=True)
    fig = px.bar(gd, x="Gender", y="learners", title="Registered Learners by Recorded Gender",
                 labels={"Gender": "Recorded gender", "learners": "Learners"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Registered learners vs learners with recorded enrollments")
    ru = len(filtered_users)
    eu = learners_with_enrollments(filtered)
    st.write(f"Registered learners (filtered): **{ru:,}**; Learners with at least one recorded enrollment (filtered): **{eu:,}**.")
    st.caption("All registered learners in this dataset have at least one recorded enrollment.")

    st.subheader("Age band × recorded gender (registered learners)")
    ct = age_gender_crosstab(filtered_users)
    st.dataframe(ct, use_container_width=True)
    fig = px.bar(ct, barmode="group", title="Registered Learners by Age Band and Gender",
                 labels={"value": "Learners", "AgeBand": "Age band"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Enrollment activity by age group and gender (transaction records)")
    e_age = enrollments_by_age_group(filtered)
    st.dataframe(e_age, use_container_width=True)
    e_gen = gender_enrollment_counts(filtered)
    st.dataframe(e_gen, use_container_width=True)

# ============================ Enrollment ===================================
with tab_enroll:
    st.subheader("Enrollments by course category")
    cp = category_popularity(filtered)
    st.dataframe(cp, use_container_width=True)
    fig = px.bar(cp.sort_values("enrollments"), x="enrollments", y="CourseCategory", orientation="h",
                 title="Course Category Popularity (enrollment records)",
                 labels={"enrollments": "Enrollment records", "CourseCategory": "Category"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Enrollments by course type")
    tp = type_popularity(filtered)
    st.dataframe(tp, use_container_width=True)
    fig = px.pie(tp, names="CourseType", values="enrollments", title="Free vs Paid Enrollment Records", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Enrollments by course level")
    lp = level_popularity(filtered)
    st.dataframe(lp, use_container_width=True)
    fig = px.bar(lp, x="CourseLevel", y="enrollments", title="Enrollment Records by Course Level",
                 labels={"CourseLevel": "Level", "enrollments": "Enrollment records"},
                 color="CourseLevel",
                 color_discrete_map={"Beginner": "#55A868", "Intermediate": "#DD8452", "Advanced": "#C44E52"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Most and least enrolled courses")
    cls = course_level_summary(filtered, courses)
    st.dataframe(cls[["CourseName", "CourseCategory", "CourseLevel", "CourseType", "enrollments", "unique_learners", "CourseRating"]].head(15), use_container_width=True)
    st.caption("Top 15 courses by enrollment records. Full table available in Course Explorer / export.")

    st.subheader("Enrollment trend over time")
    mt = monthly_trend(filtered)
    fig = px.line(mt, x="Month", y="enrollments", markers=True, title="Enrollment Records by Month")
    st.plotly_chart(fig, use_container_width=True)
    qt = quarterly_trend(filtered)
    if len(qt) > 1:
        fig2 = px.bar(qt, x="Quarter", y="enrollments", title="Enrollment Records by Quarter")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Courses per learner distribution")
    cpl = courses_per_learner(filtered)
    fig = px.histogram(cpl, nbins=max(1, int(cpl.max()) if len(cpl) else 1),
                       title="Distribution of Recorded Enrollments per Learner",
                       labels={"value": "Enrollment records per learner", "count": "Learners"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    conc = enrollment_concentration(filtered)
    st.info(f"The most active 10% of enrolled learners ({conc['top_n']:,} of {conc['total_learners']:,}) "
            f"account for **{conc['top_share_pct']:.1f}%** of enrollment records in the current filter. "
            "This describes concentration, not quality or value.")

# ============================ Cross =========================================
with tab_cross:
    st.subheader("Age group × course category")
    norm_toggle = st.radio("View", ["Counts", "Within-group percentages"], horizontal=True, key="norm1")
    normalize = norm_toggle.startswith("Within")
    hm = heatmap_age_category(filtered, normalize=normalize)
    fig = px.imshow(hm, text_auto=True, aspect="auto", color_continuous_scale="YlGnBu",
                    title="Age Band vs Course Category" + (" (% within age group)" if normalize else " (counts)"),
                    labels=dict(x="Course Category", y="Age Band", color="Enrollments" if not normalize else "%"))
    st.plotly_chart(fig, use_container_width=True)
    if normalize:
        st.caption("Percentages are within each age group (row). Counts available in Counts view.")

    st.subheader("Recorded gender × course level")
    norm2 = st.radio("View", ["Counts", "Within-group percentages"], horizontal=True, key="norm2")
    gl = gender_level_comparison(filtered, normalize=norm2.startswith("Within"))
    if norm2.startswith("Within"):
        fig = px.bar(gl, barmode="stack", title="Course Level Preference by Gender (%)",
                     labels={"value": "% of enrollment records", "Gender_Learner": "Recorded gender"})
    else:
        fig = px.bar(gl, barmode="group", title="Enrollment Records by Gender and Level",
                     labels={"value": "Enrollment records", "Gender_Learner": "Recorded gender"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age group × course level")
    al = age_level_comparison(filtered, normalize=False)
    fig = px.bar(al, barmode="group", title="Enrollment Records by Age Band and Level",
                 labels={"value": "Enrollment records", "AgeBand": "Age band"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recorded gender × course category")
    gc = gender_category_comparison(filtered, normalize=False)
    # show as heatmap
    fig = px.imshow(gc, text_auto=True, aspect="auto", color_continuous_scale="YlGnBu",
                    title="Recorded Gender vs Course Category (counts)",
                    labels=dict(x="Course Category", y="Recorded gender", color="Enrollments"))
    st.plotly_chart(fig, use_container_width=True)

    # sample size warnings
    small_groups = []
    for ag, n in filtered.groupby("AgeBand").size().items():
        if n < 30:
            small_groups.append(f"{ag}: {n} records")
    if small_groups:
        st.warning("Small groups detected (fewer than 30 enrollment records): " + "; ".join(small_groups))

# ============================ Course Explorer ==============================
with tab_course:
    st.subheader("Course-level enrollment and attributes")
    search = st.text_input("Search course name", "")
    cat_f = st.multiselect("Category", category_options, default=category_options, key="ce_cat")
    type_f = st.multiselect("Type", type_options, default=type_options, key="ce_type")
    level_f = st.multiselect("Level", level_present, default=level_present, key="ce_level")

    cls = course_level_summary(filtered, courses)
    if search:
        cls = cls[cls["CourseName"].str.contains(search, case=False, na=False)]
    cls = cls[cls["CourseCategory"].isin(cat_f) & cls["CourseType"].isin(type_f) & cls["CourseLevel"].isin(level_f)]

    show_cols = ["CourseName", "CourseCategory", "CourseLevel", "CourseType",
                 "enrollments", "unique_learners", "CoursePrice", "CourseDuration", "CourseRating"]
    show_cols = [c for c in show_cols if c in cls.columns]
    st.dataframe(cls[show_cols].sort_values("enrollments", ascending=False), use_container_width=True, height=400)

    st.caption("Aggregated course-level view only. No learner-identifying information.")

# ============================ Data Quality ================================
with tab_quality:
    st.subheader("Schema validation")
    for sheet in SHEET_NAMES:
        st.write(f"**{sheet}**: required columns present — {all(c in raw_tables[sheet].columns for c in REQUIRED_COLUMNS[sheet])}")

    st.subheader("Row counts")
    rc = pd.DataFrame([{"table": s, "rows": len(raw_tables[s]), "columns": raw_tables[s].shape[1]} for s in SHEET_NAMES])
    st.dataframe(rc, use_container_width=True)

    st.subheader("Key uniqueness")
    st.dataframe(key_uniqueness_report(raw_tables), use_container_width=True)

    st.subheader("Missing values")
    mv = missing_values_report(raw_tables)
    st.dataframe(mv, use_container_width=True)

    st.subheader("Duplicate rows")
    st.dataframe(duplicate_rows_report(raw_tables), use_container_width=True)

    st.subheader("Invalid ages")
    st.dataframe(invalid_age_report(raw_tables["Users"], raw_tables["Teachers"]), use_container_width=True)

    st.subheader("Range validation")
    st.dataframe(invalid_range_report(raw_tables["Courses"], raw_tables["Teachers"], raw_tables["Transactions"]), use_container_width=True)

    st.subheader("Referential integrity")
    st.dataframe(referential_integrity_report(raw_tables["Transactions"], raw_tables["Users"], raw_tables["Courses"], raw_tables["Teachers"]), use_container_width=True)

    st.subheader("Amount vs CoursePrice")
    st.dataframe(amount_price_discrepancy_report(raw_tables["Transactions"], raw_tables["Courses"]), use_container_width=True)

    st.subheader("Date coverage")
    st.write(f"Transaction dates range from {master['TransactionDate'].min().date()} to {master['TransactionDate'].max().date()}. "
             f"Invalid dates: {int(master['TransactionDate'].isna().sum())}.")

    st.subheader("KPI definitions")
    defs = pd.DataFrame([
        {"KPI": "Total enrollment records", "Definition": "Count of valid transaction rows after filters.", "Denominator": "Filtered transaction rows"},
        {"KPI": "Unique learner-course pairs", "Definition": "Distinct UserID+CourseID combinations.", "Denominator": "Filtered transaction rows"},
        {"KPI": "Registered learners", "Definition": "Count of users in Users sheet matching age/gender filters.", "Denominator": "Registered users"},
        {"KPI": "Learners with enrollments", "Definition": "Distinct learners in filtered enrollment records.", "Denominator": "Filtered transaction rows"},
        {"KPI": "Gender participation ratio", "Definition": "Distribution of learners by recorded gender.", "Denominator": "Registered learners"},
        {"KPI": "Category popularity index", "Definition": "Category enrollment count / total valid enrollment records.", "Denominator": "Filtered enrollment records"},
        {"KPI": "Level preference distribution", "Definition": "Enrollment records by course level.", "Denominator": "Filtered enrollment records"},
        {"KPI": "Average courses per learner", "Definition": "Average distinct courses per enrolled learner.", "Denominator": "Learners with >=1 enrollment"},
        {"KPI": "Enrollment concentration", "Definition": "Share of enrollment records from top 10% most active learners.", "Denominator": "Filtered enrollment records"},
    ])
    st.dataframe(defs, use_container_width=True)

    st.subheader("Methodology notes")
    st.markdown("""
    - One row in Transactions = one recorded enrollment/transaction event.
    - Age bands: Under 18, 18-25, 26-35, 36-45, 46 and above, Unknown/missing/invalid. (46+ avoids overlap with 36-45.)
    - Recorded gender values are preserved; no inference or recoding.
    - Left joins used for diagnostics; unmatched keys reported above.
    - No predictive modeling, monetization, or causal claims.
    - Privacy: usernames, emails, teacher names, and individual transaction histories are excluded from dashboard and default exports.
    """)

    st.subheader("Exports")
    # Category summary
    st.download_button("Download category summary (CSV)", data=to_csv_bytes(category_popularity(filtered)),
                       file_name="category_summary.csv", mime="text/csv")
    st.download_button("Download level summary (CSV)", data=to_csv_bytes(level_popularity(filtered)),
                       file_name="level_summary.csv", mime="text/csv")
    st.download_button("Download age-group summary (CSV)", data=to_csv_bytes(enrollments_by_age_group(filtered)),
                       file_name="age_group_summary.csv", mime="text/csv")
    st.download_button("Download gender summary (CSV)", data=to_csv_bytes(gender_enrollment_counts(filtered)),
                       file_name="gender_summary.csv", mime="text/csv")
    st.download_button("Download course summary (CSV)", data=to_csv_bytes(course_level_summary(filtered, courses)),
                       file_name="course_summary.csv", mime="text/csv")
    st.download_button("Download data-quality summary (CSV)",
                       data=to_csv_bytes(referential_integrity_report(raw_tables["Transactions"], raw_tables["Users"], raw_tables["Courses"], raw_tables["Teachers"])),
                       file_name="data_quality_summary.csv", mime="text/csv")
    filters = {
        "age_group": age_sel, "gender": gender_sel, "category": category_sel,
        "type": type_sel, "level": level_sel, "date_range": date_range,
    }
    st.download_button("Download active filter selections (TXT)",
                       data=filter_selections_text(filters).encode("utf-8"),
                       file_name="filter_selections.txt", mime="text/plain")

st.markdown("---")
st.caption("EduPro Learner Demographics and Course Enrollment Behavior Analysis — descriptive analytics only.")
