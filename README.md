# Management Report App

Modern Streamlit app for management sales reporting from CSV uploads.

## Features
- Filters only `type = Mobile device`.
- Merges dealership names by text before the first comma.
  - Example: `ABC Motors, North Zone` + `ABC Motors, East Zone` => `ABC Motors`
- Date filtering using `disbursedon_date`:
  - All Dates
  - This Week
  - This Month
  - Custom Range
- Report tables:
  - Sales by Dealership
  - Sales by Model
  - Sales by Branch
  - Sales by Dealership + Model + Branch
  - Sales by Month
  - Sales by Week (Monday start)
- Export options:
  - CSV per report
  - Full management report as one Excel file (multi-sheet)
  - Filtered raw data CSV

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

The app normalizes header formatting (spaces/underscores/case), so small differences are accepted.

## Run locally
```bash
cd /Users/dembsdesign/Documents/management-report-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
