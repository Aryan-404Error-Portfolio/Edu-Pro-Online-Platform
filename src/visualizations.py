"""Plotly chart helpers for EduPro dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLORBLIND = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def bar_chart(x, y, title, orientation="v", labels=None, color=None):
    if orientation == "h":
        fig = px.bar(x=x, y=y, orientation="h", title=title, labels=labels or {})
    else:
        fig = px.bar(x=x, y=y, title=title, labels=labels or {})
    if color:
        fig.update_traces(marker_color=color)
    return fig


def pie_chart(names, values, title, hole=0.4):
    return px.pie(names=names, values=values, title=title, hole=hole, color_discrete_sequence=COLORBLIND)


def heatmap(df, title, labels=None):
    fig = px.imshow(df, text_auto=True, aspect="auto", color_continuous_scale="YlGnBu", title=title)
    if labels:
        fig.update_layout(xaxis_title=labels.get("x"), yaxis_title=labels.get("y"))
    return fig


def line_chart(df, x, y, title):
    return px.line(df, x=x, y=y, markers=True, title=title)
