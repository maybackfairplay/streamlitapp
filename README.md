# Management Report App

This app lets you upload a CSV and generate management sales reports.

## Rules applied
- Counts only rows where `type` is `Mobile device`.
- Merges dealership names by using the text before the first comma.
  - Example: `ABC Motors, North Zone` and `ABC Motors, East Zone` are counted as `ABC Motors`.
- Supports report period filters using `disbursedon_date`:
  - All Dates
  - This Week
  - This Month
  - Custom Range
- Shows sales by:
  - Dealership
  - Model
  - Branch
  - Dealership + Model + Branch (detailed)
  - Month
  - Week (Monday start)

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
