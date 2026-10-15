"""Privacy-conscious CSV export helpers."""

from __future__ import annotations

import io

import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def filter_selections_text(filters: dict) -> str:
    lines = ["EduPro filter selections"]
    for k, v in filters.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines) + "\n"


def make_summary_export(df: pd.DataFrame, name: str) -> bytes:
    return to_csv_bytes(df)
