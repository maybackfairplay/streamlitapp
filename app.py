import os
from datetime import datetime
from io import BytesIO
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from reporting import (
    apply_date_filter,
    build_reports,
    compare_periods,
    data_quality_summary,
    filter_data,
    load_snapshot_df,
    load_upload_history,
    map_columns,
    prepare_data,
    save_upload_snapshot,
    send_email_report,
    to_csv_download,
    to_excel_download,
    to_pdf_download,
    validate_required_columns,
)

st.set_page_config(page_title="Management Sales Report", layout="wide")


def get_auth_config() -> Dict[str, Dict[str, str]]:
    users = {
        "admin": {"password": "admin123", "role": "admin"},
        "viewer": {"password": "viewer123", "role": "viewer"},
    }
    try:
        if "auth_users" in st.secrets:
            return dict(st.secrets["auth_users"])
    except Exception:
        pass

    env_user = os.getenv("APP_USER")
    env_pass = os.getenv("APP_PASSWORD")
    env_role = os.getenv("APP_ROLE", "admin")
    if env_user and env_pass:
        users = {env_user: {"password": env_pass, "role": env_role}}
    return users


def login_ui() -> Optional[str]:
    users = get_auth_config()
    if st.session_state.get("authenticated"):
        return st.session_state.get("role")

    st.title("Management Sales Report")
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        entry = users.get(username)
        if entry and password == entry.get("password"):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["role"] = entry.get("role", "viewer")
            st.rerun()
        st.error("Invalid username or password")

    st.caption("Default demo credentials: admin/admin123 or viewer/viewer123")
    return None


@st.cache_data(show_spinner=False)
def parse_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes), dtype=str)


@st.cache_data(show_spinner=False)
def process_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    prepared = prepare_data(df)
    reports = build_reports(prepared)
    return prepared, reports


def show_report_section(title: str, report_key: str, reports: Dict[str, pd.DataFrame], filename: str) -> None:
    st.subheader(title)
    st.dataframe(reports[report_key], use_container_width=True, height=360)
    st.download_button(
        f"Download {title} (CSV)",
        data=to_csv_download(reports[report_key]),
        file_name=filename,
        mime="text/csv",
    )


role = login_ui()
if role is None:
    st.stop()

st.title("Management Sales Report")
st.caption("Upload CSV and generate dealership, branch, model, trend, and export reports")

with st.sidebar:
    st.markdown("### Session")
    st.write(f"User: `{st.session_state.get('username')}`")
    st.write(f"Role: `{role}`")
    if st.button("Logout", use_container_width=True):
        for k in ["authenticated", "username", "role", "uploaded_source"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown("### Upload History")
    history = load_upload_history()
    options = [f"{h['timestamp']} | {h['original_name']} ({h['rows']} rows)" for h in history]
    selected_idx = st.selectbox("Reopen previous upload", options=(["None"] + options), index=0)
    if selected_idx != "None":
        selected = history[options.index(selected_idx)]
        if st.button("Load Selected", use_container_width=True):
            st.session_state["uploaded_source"] = {"saved_name": selected["saved_name"], "name": selected["original_name"]}
            st.rerun()

st.markdown("---")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

raw_df: Optional[pd.DataFrame] = None
source_name = ""

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    with st.status("Processing upload...", expanded=False) as status:
        status.write("Parsing CSV")
        raw_df = parse_csv_bytes(file_bytes)
        status.write("Saving upload history")
        save_upload_snapshot(uploaded_file.name, file_bytes)
        st.session_state["uploaded_source"] = {"saved_name": None, "name": uploaded_file.name, "bytes": file_bytes}
        status.update(label="Upload ready", state="complete")
    source_name = uploaded_file.name
elif st.session_state.get("uploaded_source"):
    src = st.session_state["uploaded_source"]
    source_name = src.get("name", "restored.csv")
    try:
        if src.get("saved_name"):
            raw_df = load_snapshot_df(src["saved_name"])
        elif src.get("bytes"):
            raw_df = parse_csv_bytes(src["bytes"])
    except Exception as exc:
        st.error(f"Failed to load saved upload: {exc}")

if raw_df is None:
    st.info("Upload a CSV or load one from Upload History.")
    st.stop()

st.caption(f"Loaded source: {source_name}")

df = map_columns(raw_df)
missing_columns = validate_required_columns(df)
if missing_columns:
    st.error(
        "Missing required columns: " + ", ".join(missing_columns) + "\n"
        "Expected source headers: Client Mobile No, Branch Office, Dealership, disbursedon_date, registration_no, chasis_no, make, model, type"
    )
    st.stop()

quality = data_quality_summary(df)
st.subheader("Data Quality")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Total Rows", quality.total_rows)
q2.metric("Mobile Device Rows", quality.mobile_rows)
q3.metric("Excluded Non-Mobile", quality.excluded_non_mobile)
q4.metric("Duplicate Rows", quality.duplicate_rows)
q5, q6, q7 = st.columns(3)
q5.metric("Invalid Dates", quality.invalid_dates)
q6.metric("Missing Dealership", quality.missing_dealership)
q7.metric("Missing Branch/Model", quality.missing_branch + quality.missing_model)

with st.status("Preparing analytics...", expanded=False) as status:
    status.write("Applying rules")
    prepared, _ = process_dataframe(df)
    status.update(label="Ready", state="complete")

if prepared.empty:
    st.warning("No rows found where type is 'Mobile device'.")
    st.stop()

st.subheader("Filters")
has_valid_dates = prepared["disbursed_date"].notna().any()
col1, col2, col3 = st.columns(3)
filter_mode = col1.selectbox("Report Period", ["All Dates", "This Week", "This Month", "Custom Range"])

all_dealers = sorted([x for x in prepared["dealership"].dropna().unique().tolist() if str(x).strip()])
all_branches = sorted([x for x in prepared["branch_office"].dropna().unique().tolist() if str(x).strip()])
all_models = sorted([x for x in prepared["model"].dropna().unique().tolist() if str(x).strip()])
all_makes = sorted([x for x in prepared["make"].dropna().unique().tolist() if str(x).strip()])

sel_dealers = col2.multiselect("Dealership", options=all_dealers)
sel_branches = col3.multiselect("Branch", options=all_branches)

col4, col5 = st.columns(2)
sel_models = col4.multiselect("Model", options=all_models)
sel_makes = col5.multiselect("Make", options=all_makes)

start_dt = None
end_dt = None
if filter_mode == "Custom Range":
    if has_valid_dates:
        min_date = prepared["disbursed_date"].min().date()
        max_date = prepared["disbursed_date"].max().date()
        selected_range = st.date_input("Custom Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_dt, end_dt = selected_range
        else:
            st.error("Please select both start and end dates.")
            st.stop()
    else:
        st.warning("No valid dates found. Date filter disabled.")
        filter_mode = "All Dates"

if filter_mode in {"This Week", "This Month"} and not has_valid_dates:
    st.warning("No valid dates found. Showing all dates.")
    filter_mode = "All Dates"

filtered = filter_data(prepared, sel_dealers, sel_branches, sel_models, sel_makes)
filtered = apply_date_filter(filtered, filter_mode, start_dt, end_dt)

if filtered.empty:
    st.warning("No records match selected filters.")
    st.stop()

reports = build_reports(filtered)
comparison = compare_periods(filtered)

st.subheader("Snapshot")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Units", int(len(filtered)))
m2.metric("Dealerships", int(reports["sales_by_dealership"].shape[0]))
m3.metric("Branches", int(reports["sales_by_branch"].shape[0]))
m4.metric("Models", int(reports["sales_by_model"].shape[0]))
m5.metric("Period Change", f"{comparison['pct_change']:.1f}%")

st.subheader("Trends")
ch1, ch2 = st.columns(2)
with ch1:
    if "sales_by_week" in reports:
        week_df = reports["sales_by_week"].rename(columns={"week_start": "period"}).set_index("period")
        st.line_chart(week_df["sales_count"])
    else:
        st.info("Weekly trend unavailable (no valid dates).")
with ch2:
    branch_top = reports["sales_by_branch"].head(10).set_index("branch_office")
    st.bar_chart(branch_top["sales_count"])

st.subheader("Exports")
excel_bytes = to_excel_download(reports)
pdf_bytes = to_pdf_download(reports, comparison, int(len(filtered)))

ex1, ex2, ex3 = st.columns(3)
ex1.download_button(
    "Export Full Report (Excel)",
    data=excel_bytes,
    file_name="management_sales_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
ex2.download_button(
    "Export Filtered Raw Data (CSV)",
    data=to_csv_download(filtered),
    file_name="filtered_mobile_device_data.csv",
    mime="text/csv",
)
ex3.download_button(
    "Export Briefing (PDF)",
    data=pdf_bytes,
    file_name="management_briefing.pdf",
    mime="application/pdf",
)

if role == "admin":
    st.subheader("Email Report (Admin)")
    recipients_text = st.text_input("Recipients (comma-separated)", value="")
    email_subject = st.text_input("Subject", value=f"Management Sales Report - {datetime.now().strftime('%Y-%m-%d')}")

    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("SMTP_SENDER", smtp_user)

    if st.button("Send Report Email"):
        recipients = [x.strip() for x in recipients_text.split(",") if x.strip()]
        ok, msg = send_email_report(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            sender=sender,
            recipients=recipients,
            subject=email_subject,
            body="Attached is the latest management sales report.",
            attachment_name="management_sales_report.xlsx",
            attachment_bytes=excel_bytes,
        )
        if ok:
            st.success(msg)
        else:
            st.error(msg + " | Configure SMTP_* env vars")

st.subheader("Reports")
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
    st.subheader("Time Reports")
    t1, t2 = st.columns(2)
    if "sales_by_month" in reports:
        with t1:
            show_report_section("Sales by Month", "sales_by_month", reports, "sales_by_month.csv")
    if "sales_by_week" in reports:
        with t2:
            show_report_section("Sales by Week", "sales_by_week", reports, "sales_by_week.csv")
