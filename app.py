import re
from io import BytesIO, StringIO
from typing import Dict, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Management Sales Report", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
      background: radial-gradient(circle at 10% 10%, #eaf4ff 0%, #f6f9fc 45%, #ffffff 100%);
    }
    .hero {
      padding: 1rem 1.25rem;
      border-radius: 14px;
      background: linear-gradient(135deg, #16324f 0%, #234f7d 40%, #2f6aa1 100%);
      color: #ffffff;
      box-shadow: 0 10px 30px rgba(18, 34, 52, 0.2);
      margin-bottom: 1rem;
    }
    .hero h1 {
      margin: 0;
      font-size: 1.8rem;
      font-weight: 700;
    }
    .hero p {
      margin: 0.35rem 0 0 0;
      opacity: 0.95;
      font-size: 0.95rem;
    }
    .card {
      border: 1px solid #d8e5f3;
      border-radius: 12px;
      padding: 0.8rem 1rem;
      background: #ffffff;
      box-shadow: 0 4px 12px rgba(13, 39, 68, 0.06);
    }
    div[data-testid="stMetric"] {
      background: #ffffff;
      border: 1px solid #d8e5f3;
      border-radius: 12px;
      padding: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


CANONICAL_COLUMNS = {
    "client_mobile_no": "clientmobileno",
    "branch_office": "branchoffice",
    "dealership": "dealership",
    "disbursedon_date": "disbursedondate",
    "registration_no": "registrationno",
    "chasis_no": "chasisno",
    "make": "make",
    "model": "model",
    "type": "type",
}

REQUIRED_FOR_REPORT = ["branch_office", "dealership", "model", "type"]


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_source = {normalize_col_name(c): c for c in df.columns}
    rename_map = {}

    for canonical, normalized in CANONICAL_COLUMNS.items():
        if normalized in normalized_source:
            rename_map[normalized_source[normalized]] = canonical

    return df.rename(columns=rename_map)


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_FOR_REPORT if c not in df.columns]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["type"] = df["type"].astype(str).str.strip()
    df = df[df["type"].str.lower() == "mobile device"]

    dealership_base = (
        df["dealership"].astype(str).fillna("").str.split(",").str[0].str.strip()
    )
    df["dealership_key"] = dealership_base.str.lower()
    df["dealership"] = dealership_base

    dealership_name_map = (
        df[df["dealership_key"] != ""]
        .drop_duplicates(subset=["dealership_key"])
        .set_index("dealership_key")["dealership"]
        .to_dict()
    )
    df["dealership"] = df["dealership_key"].map(dealership_name_map).fillna(df["dealership"])

    df["branch_office"] = df["branch_office"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df["disbursed_date"] = pd.to_datetime(df.get("disbursedon_date"), errors="coerce")

    return df


def build_reports(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    reports = {
        "sales_by_dealership": (
            df.groupby("dealership", dropna=False)
            .size()
            .reset_index(name="sales_count")
            .sort_values("sales_count", ascending=False)
        ),
        "sales_by_model": (
            df.groupby("model", dropna=False)
            .size()
            .reset_index(name="sales_count")
            .sort_values("sales_count", ascending=False)
        ),
        "sales_by_branch": (
            df.groupby("branch_office", dropna=False)
            .size()
            .reset_index(name="sales_count")
            .sort_values("sales_count", ascending=False)
        ),
        "dealership_model_branch": (
            df.groupby(["dealership", "model", "branch_office"], dropna=False)
            .size()
            .reset_index(name="sales_count")
            .sort_values("sales_count", ascending=False)
        ),
    }

    dated_df = df[df["disbursed_date"].notna()].copy()
    if not dated_df.empty:
        dated_df["month"] = dated_df["disbursed_date"].dt.to_period("M").astype(str)
        dated_df["week_start"] = (
            dated_df["disbursed_date"]
            - pd.to_timedelta(dated_df["disbursed_date"].dt.dayofweek, unit="D")
        ).dt.date.astype(str)

        reports["sales_by_month"] = (
            dated_df.groupby("month", dropna=False)
            .size()
            .reset_index(name="sales_count")
            .sort_values("month", ascending=False)
        )
        reports["sales_by_week"] = (
            dated_df.groupby("week_start", dropna=False)
            .size()
            .reset_index(name="sales_count")
            .sort_values("week_start", ascending=False)
        )

    return reports


def apply_date_filter(
    df: pd.DataFrame,
    filter_mode: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> pd.DataFrame:
    if filter_mode == "All Dates":
        return df

    dated_only = df[df["disbursed_date"].notna()].copy()
    if dated_only.empty:
        return dated_only

    if filter_mode == "This Week":
        today = pd.Timestamp.today().normalize()
        start = today - pd.Timedelta(days=today.dayofweek)
        end = start + pd.Timedelta(days=6)
    elif filter_mode == "This Month":
        today = pd.Timestamp.today().normalize()
        start = today.replace(day=1)
        end = (start + pd.offsets.MonthEnd(1)).normalize()
    else:
        if start_date is None or end_date is None:
            return dated_only
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()

    return dated_only[
        (dated_only["disbursed_date"] >= start)
        & (dated_only["disbursed_date"] <= end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    ]


def to_csv_download(df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def to_excel_download(reports: Dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    sheet_map = {
        "sales_by_dealership": "By_Dealership",
        "sales_by_model": "By_Model",
        "sales_by_branch": "By_Branch",
        "dealership_model_branch": "Detail",
        "sales_by_month": "By_Month",
        "sales_by_week": "By_Week",
    }

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for key, sheet_name in sheet_map.items():
            if key in reports:
                reports[key].to_excel(writer, sheet_name=sheet_name, index=False)

    return buffer.getvalue()


def show_report_section(
    title: str, report_key: str, reports: Dict[str, pd.DataFrame], filename: str
) -> None:
    st.markdown(f"### {title}")
    st.dataframe(reports[report_key], use_container_width=True, height=320)
    st.download_button(
        f"Download {title} (CSV)",
        data=to_csv_download(reports[report_key]),
        file_name=filename,
        mime="text/csv",
    )


st.markdown(
    """
    <div class="hero">
      <h1>Management Sales Report</h1>
      <p>Upload CSV, filter by date, and export management-ready dealership, model, branch, weekly, and monthly summaries.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.info("Upload a CSV file to generate reports.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file, dtype=str)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

df = map_columns(raw_df)
missing_columns = validate_required_columns(df)
if missing_columns:
    st.error(
        "Missing required columns: " + ", ".join(missing_columns) + "\n"
        "Expected source headers: Client Mobile No, Branch Office, Dealership, "
        "disbursedon_date, registration_no, chasis_no, make, model, type"
    )
    st.stop()

prepared = prepare_data(df)
if prepared.empty:
    st.warning("No rows found where type is 'Mobile device'.")
    st.stop()

has_valid_dates = prepared["disbursed_date"].notna().any()
st.markdown("### Filters")
left, right = st.columns([2, 3])
filter_mode = left.selectbox("Report Period", ["All Dates", "This Week", "This Month", "Custom Range"])

start_dt = None
end_dt = None
if filter_mode == "Custom Range":
    if has_valid_dates:
        min_date = prepared["disbursed_date"].min().date()
        max_date = prepared["disbursed_date"].max().date()
        selected_range = right.date_input(
            "Custom Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_dt, end_dt = selected_range
        else:
            st.error("Please select both start and end dates.")
            st.stop()
        if start_dt > end_dt:
            st.error("Start Date cannot be after End Date.")
            st.stop()
    else:
        st.warning("No valid `disbursedon_date` values found. Date filter cannot be applied.")
        filter_mode = "All Dates"

if filter_mode in {"This Week", "This Month"} and not has_valid_dates:
    st.warning("No valid `disbursedon_date` values found. Showing all dates.")
    filter_mode = "All Dates"

filtered = apply_date_filter(prepared, filter_mode, start_dt, end_dt)
if filtered.empty:
    st.warning("No records match the selected date filter.")
    st.stop()

reports = build_reports(filtered)

st.markdown("### Snapshot")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Mobile Device Sales", int(len(filtered)))
m2.metric("Dealerships", int(reports["sales_by_dealership"].shape[0]))
m3.metric("Branches", int(reports["sales_by_branch"].shape[0]))
m4.metric("Models", int(reports["sales_by_model"].shape[0]))

st.markdown("### Export")
col_export1, col_export2 = st.columns([1, 1])
col_export1.download_button(
    "Export Full Report (Excel)",
    data=to_excel_download(reports),
    file_name="management_sales_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
col_export2.download_button(
    "Export Filtered Raw Data (CSV)",
    data=to_csv_download(filtered),
    file_name="filtered_mobile_device_data.csv",
    mime="text/csv",
)

tab1, tab2, tab3, tab4 = st.tabs(["Dealership", "Model", "Branch", "Detailed"])
with tab1:
    show_report_section("Sales by Dealership", "sales_by_dealership", reports, "sales_by_dealership.csv")
with tab2:
    show_report_section("Sales by Model", "sales_by_model", reports, "sales_by_model.csv")
with tab3:
    show_report_section("Sales by Branch", "sales_by_branch", reports, "sales_by_branch.csv")
with tab4:
    show_report_section(
        "Sales by Dealership + Model + Branch",
        "dealership_model_branch",
        reports,
        "sales_by_dealership_model_branch.csv",
    )

if "sales_by_month" in reports or "sales_by_week" in reports:
    st.markdown("### Time Trends")
    time_col1, time_col2 = st.columns(2)
    if "sales_by_month" in reports:
        with time_col1:
            show_report_section("Sales by Month", "sales_by_month", reports, "sales_by_month.csv")
    if "sales_by_week" in reports:
        with time_col2:
            show_report_section("Sales by Week", "sales_by_week", reports, "sales_by_week.csv")
