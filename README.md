# EduPro — Learner Demographics and Course Enrollment Behavior Analysis

Descriptive analytics dashboard for EduPro's learner base and course enrollment behavior.
Built with Python, pandas, Streamlit, and Plotly.

## Project structure

```
Edu-Pro-Online-Platform/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── README.md
├── data/                   # Raw source files (untouched)
│   ├── EduPro_Online_Platform_xlsx_-_Users.csv
│   ├── EduPro_Online_Platform_xlsx_-_Courses.csv
│   ├── EduPro_Online_Platform_xlsx_-_Transactions.csv
│   └── EduPro_Online_Platform_xlsx_-_Teachers.csv
├── src/
│   ├── data_loader.py      # Workbook/CSV loading and schema validation
│   ├── cleaning.py         # Cleaning, age bands, referential integrity, DQ reports
│   ├── metrics.py          # KPI and aggregation logic
│   ├── visualizations.py   # Chart helpers
│   └── exports.py          # Privacy-conscious CSV exports
└── tests/                  # pytest tests
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the dashboard

```bash
streamlit run app.py
```

The app reads from `data/`. If an Excel workbook `EduPro Online Platform.xlsx` is present in `data/`,
it will be loaded by sheet name; otherwise the CSV files are used.

## Run tests

```bash
pytest -q
```

## Privacy

Usernames, emails, teacher names, and individual transaction histories are excluded from the
dashboard and default exports. Only aggregated tables are displayed and exported.

## Methodology

- One row in Transactions = one recorded enrollment/transaction event.
- Age bands: Under 18, 18-25, 26-35, 36-45, 46 and above, Unknown/missing/invalid.
- Recorded gender values are preserved; no inference or recoding.
- Descriptive analytics only: no prediction, monetization, or causal claims.
PY