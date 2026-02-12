import re
from io import StringIO

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Management Sales Report", layout="wide")


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

    # Keep only rows where type is Mobile device
    df["type"] = df["type"].astype(str).str.strip()
    df = df[df["type"].str.lower() == "mobile device"]

    # Merge dealer names by text before comma, case-insensitive
    dealership_base = (
        df["dealership"].astype(str).fillna("").str.split(",").str[0].str.strip()
    )

    df["dealership_key"] = dealership_base.str.lower()
    df["dealership"] = dealership_base

    # Preserve first-seen display text while grouping case-insensitively
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


def build_reports(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sales_by_dealership = (
        df.groupby("dealership", dropna=False)
        .size()
        .reset_index(name="sales_count")
        .sort_values("sales_count", ascending=False)
    )

    sales_by_model = (
        df.groupby("model", dropna=False)
        .size()
        .reset_index(name="sales_count")
        .sort_values("sales_count", ascending=False)
    )

    sales_by_branch = (
        df.groupby("branch_office", dropna=False)
        .size()
        .reset_index(name="sales_count")
        .sort_values("sales_count", ascending=False)
    )

    dealership_model_branch = (
        df.groupby(["dealership", "model", "branch_office"], dropna=False)
        .size()
        .reset_index(name="sales_count")
        .sort_values("sales_count", ascending=False)
    )

    reports = {
        "sales_by_dealership": sales_by_dealership,
        "sales_by_model": sales_by_model,
        "sales_by_branch": sales_by_branch,
        "dealership_model_branch": dealership_model_branch,
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
    df: pd.DataFrame, filter_mode: str, start_date: pd.Timestamp | None, end_date: pd.Timestamp | None
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


st.title("Management Sales Report")
st.write(
    "Upload your CSV and get sales summaries by dealership, model, and branch. "
    "Only rows with `type = Mobile device` are counted."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
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
            "Expected source headers include: Client Mobile No, Branch Office, Dealership, "
            "disbursedon_date, registration_no, chasis_no, make, model, type"
        )
        st.stop()

    prepared = prepare_data(df)

    if prepared.empty:
        st.warning("No rows found where type is 'Mobile device'.")
        st.stop()

    has_valid_dates = prepared["disbursed_date"].notna().any()

    st.subheader("Date Filter")
    filter_mode = st.selectbox("Report Period", ["All Dates", "This Week", "This Month", "Custom Range"])

    start_dt = None
    end_dt = None
    if filter_mode == "Custom Range":
        if has_valid_dates:
            min_date = prepared["disbursed_date"].min().date()
            max_date = prepared["disbursed_date"].max().date()
            c1, c2 = st.columns(2)
            start_dt = c1.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            end_dt = c2.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
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

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Mobile Device Sales", int(len(filtered)))
    c2.metric("Unique Dealerships", int(reports["sales_by_dealership"].shape[0]))
    c3.metric("Unique Branches", int(reports["sales_by_branch"].shape[0]))

    st.subheader("Sales by Dealership")
    st.dataframe(reports["sales_by_dealership"], use_container_width=True)
    st.download_button(
        "Download Sales by Dealership (CSV)",
        data=to_csv_download(reports["sales_by_dealership"]),
        file_name="sales_by_dealership.csv",
        mime="text/csv",
    )

    st.subheader("Sales by Model")
    st.dataframe(reports["sales_by_model"], use_container_width=True)
    st.download_button(
        "Download Sales by Model (CSV)",
        data=to_csv_download(reports["sales_by_model"]),
        file_name="sales_by_model.csv",
        mime="text/csv",
    )

    st.subheader("Sales by Branch")
    st.dataframe(reports["sales_by_branch"], use_container_width=True)
    st.download_button(
        "Download Sales by Branch (CSV)",
        data=to_csv_download(reports["sales_by_branch"]),
        file_name="sales_by_branch.csv",
        mime="text/csv",
    )

    st.subheader("Sales by Dealership + Model + Branch")
    st.dataframe(reports["dealership_model_branch"], use_container_width=True)
    st.download_button(
        "Download Detailed Sales (CSV)",
        data=to_csv_download(reports["dealership_model_branch"]),
        file_name="sales_by_dealership_model_branch.csv",
        mime="text/csv",
    )

    if "sales_by_month" in reports:
        st.subheader("Sales by Month")
        st.dataframe(reports["sales_by_month"], use_container_width=True)
        st.download_button(
            "Download Sales by Month (CSV)",
            data=to_csv_download(reports["sales_by_month"]),
            file_name="sales_by_month.csv",
            mime="text/csv",
        )

    if "sales_by_week" in reports:
        st.subheader("Sales by Week")
        st.dataframe(reports["sales_by_week"], use_container_width=True)
        st.download_button(
            "Download Sales by Week (CSV)",
            data=to_csv_download(reports["sales_by_week"]),
            file_name="sales_by_week.csv",
            mime="text/csv",
        )
else:
    st.info("Upload a CSV file to generate the report.")
