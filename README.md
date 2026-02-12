# Management Report App

Cyber-styled Streamlit app for management sales reporting from CSV uploads.

## Implemented Enhancements
1. Data quality checks panel (invalid dates, missing values, duplicates, excluded rows)
2. Drill-down filters (dealership, branch, model, make + date period)
3. Trend visuals (weekly line + top-branch bar)
4. Comparison mode (current vs previous period KPI)
5. Scheduled email dispatch support (SMTP env vars)
6. PDF briefing export
7. Upload history with reopen capability
8. Performance/reliability improvements (`st.cache_data`, status feedback, safer errors)
9. Security and access (login + role-based control: `admin`, `viewer`)
10. Deployment hardening (pinned dependencies + sample CSV + tests)

## AI Capabilities
- Auto insights and root-cause suggestions
- Weekly anomaly detection
- Forecasting by dealership/model
- Natural-language Q&A
- Smart alerts and risk notifications
- Data cleaning assistant suggestions
- Target-vs-actual analysis + AI target recommendations
- Narrative briefing generator
- What-if simulator
- Scheduled report helper
- Saved filter presets
- Audit trail logging

## Login
Default credentials (change via env/secrets):
- `admin / admin123`
- `viewer / viewer123`

## SMTP (for email reports)
Set environment variables:
- `SMTP_HOST`
- `SMTP_PORT` (default `587`)
- `SMTP_USER`
- `SMTP_PASSWORD`
- `SMTP_SENDER`

## Expected source columns
- `Client Mobile No`
- `Branch Office`
- `Dealership`
- `disbursedon_date`
- `registration_no`
- `chasis_no`
- `make`
- `model`
- `type`

## Run locally
```bash
cd /Users/dembsdesign/Documents/management-report-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Run tests
```bash
pip install -r requirements-dev.txt
pytest
```
