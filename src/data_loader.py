"""Workbook/CSV ingestion and schema validation for EduPro."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

SHEET_NAMES = ["Users", "Courses", "Transactions", "Teachers"]

REQUIRED_COLUMNS: Dict[str, list] = {
    "Users": ["UserID", "UserName", "Age", "Gender", "Email"],
    "Courses": [
        "CourseID", "CourseName", "CourseCategory", "CourseType", "CourseLevel",
        "CoursePrice", "CourseDuration", "CourseRating",
    ],
    "Transactions": [
        "TransactionID", "UserID", "CourseID", "TransactionDate", "Amount",
        "PaymentMethod", "TeacherID",
    ],
    "Teachers": [
        "TeacherID", "TeacherName", "Age", "Gender", "Expertise",
        "YearsOfExperience", "TeacherRating",
    ],
}

CSV_FILENAMES = {
    "Users": "EduPro_Online_Platform_xlsx_-_Users.csv",
    "Courses": "EduPro_Online_Platform_xlsx_-_Courses.csv",
    "Transactions": "EduPro_Online_Platform_xlsx_-_Transactions.csv",
    "Teachers": "EduPro_Online_Platform_xlsx_-_Teachers.csv",
}

XLSX_FILENAME = "EduPro Online Platform.xlsx"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_data_dir(data_dir=None) -> Path:
    if data_dir is None:
        return _project_root() / "data"
    return Path(data_dir)


def load_raw_tables(data_dir=None) -> Dict[str, pd.DataFrame]:
    """Load the four sheets/CSVs and validate required columns."""
    ddir = resolve_data_dir(data_dir)
    tables: Dict[str, pd.DataFrame] = {}

    xlsx_path = ddir / XLSX_FILENAME
    if xlsx_path.exists():
        try:
            xl = pd.read_excel(xlsx_path, sheet_name=None)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(f"Could not read Excel workbook {xlsx_path}: {exc}") from exc
        for sheet in SHEET_NAMES:
            if sheet not in xl:
                raise ValueError(f"Required sheet '{sheet}' not found in workbook. Found: {list(xl.keys())}")
            tables[sheet] = xl[sheet].copy()
    else:
        for sheet in SHEET_NAMES:
            csv_path = ddir / CSV_FILENAMES[sheet]
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Required data file not found: {csv_path}. "
                    f"Place the workbook CSVs in {ddir}."
                )
            tables[sheet] = pd.read_csv(csv_path)

    for sheet, df in tables.items():
        missing = [c for c in REQUIRED_COLUMNS[sheet] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Sheet '{sheet}' is missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )
    return tables


def normalize_types(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Normalize dtypes without dropping rows."""
    t = {k: v.copy() for k, v in tables.items()}

    # Users
    u = t["Users"]
    u["UserID"] = u["UserID"].astype(str).str.strip()
    u["UserName"] = u["UserName"].astype(str)
    u["Age"] = pd.to_numeric(u["Age"], errors="coerce")
    u["Gender"] = u["Gender"].astype(str).str.strip()
    u.loc[u["Gender"].isin(["", "nan", "None", "NaN"]), "Gender"] = "Unknown / missing"
    u["Email"] = u["Email"].astype(str)

    # Courses
    c = t["Courses"]
    c["CourseID"] = c["CourseID"].astype(str).str.strip()
    c["CourseName"] = c["CourseName"].astype(str)
    c["CourseCategory"] = c["CourseCategory"].astype(str).str.strip()
    c["CourseType"] = c["CourseType"].astype(str).str.strip()
    c["CourseLevel"] = c["CourseLevel"].astype(str).str.strip()
    for col in ["CoursePrice", "CourseDuration", "CourseRating"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")

    # Transactions
    tr = t["Transactions"]
    tr["TransactionID"] = tr["TransactionID"].astype(str).str.strip()
    tr["UserID"] = tr["UserID"].astype(str).str.strip()
    tr["CourseID"] = tr["CourseID"].astype(str).str.strip()
    tr["TransactionDate"] = pd.to_datetime(tr["TransactionDate"], errors="coerce", dayfirst=True)
    tr["Amount"] = pd.to_numeric(tr["Amount"], errors="coerce")
    tr["PaymentMethod"] = tr["PaymentMethod"].astype(str).str.strip()
    tr["TeacherID"] = tr["TeacherID"].astype(str).str.strip()

    # Teachers
    te = t["Teachers"]
    te["TeacherID"] = te["TeacherID"].astype(str).str.strip()
    te["TeacherName"] = te["TeacherName"].astype(str)
    te["Age"] = pd.to_numeric(te["Age"], errors="coerce")
    te["Gender"] = te["Gender"].astype(str).str.strip()
    te.loc[te["Gender"].isin(["", "nan", "None", "NaN"]), "Gender"] = "Unknown / missing"
    te["Expertise"] = te["Expertise"].astype(str).str.strip()
    te["YearsOfExperience"] = pd.to_numeric(te["YearsOfExperience"], errors="coerce")
    te["TeacherRating"] = pd.to_numeric(te["TeacherRating"], errors="coerce")

    return t


def load_data(data_dir=None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load and normalize; return (raw_tables, normalized_tables)."""
    raw = load_raw_tables(data_dir)
    norm = normalize_types(raw)
    return raw, norm
