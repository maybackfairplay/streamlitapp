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

st.set_page_config(page_title="WATU Daily Sales", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Rajdhani:wght@500;700&display=swap');
    :root {
      --bg0: #030916;
      --bg1: #061127;
      --line: rgba(58, 87, 124, 0.25);
      --text: #dbe8ff;
      --muted: #7f93b6;
      --cyan: #00d9e8;
      --lime: #9cff3a;
      --pink: #ff2e73;
      --amber: #ffbf3d;
    }
    .stApp {
      color: var(--text);
      background:
        linear-gradient(var(--line) 1px, transparent 1px),
        linear-gradient(90deg, var(--line) 1px, transparent 1px),
        radial-gradient(circle at 20% -10%, #0f2649 0%, transparent 35%),
        radial-gradient(circle at 90% 10%, #06203d 0%, transparent 35%),
        linear-gradient(180deg, var(--bg1), var(--bg0));
      background-size: 42px 42px, 42px 42px, auto, auto, auto;
      font-family: "Rajdhani", sans-serif;
      position: relative;
      overflow-x: hidden;
    }
    .stApp::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(180deg, rgba(255,255,255,0.015) 0%, rgba(255,255,255,0.015) 48%, rgba(0,0,0,0.0) 49%, rgba(0,0,0,0.0) 100%);
      background-size: 100% 5px;
      mix-blend-mode: screen;
      opacity: 0.24;
      animation: scan 7s linear infinite;
      z-index: 0;
    }
    @keyframes scan {0% {transform: translateY(-4px);} 100% {transform: translateY(4px);}}
    #MainMenu, header[data-testid="stHeader"], footer {visibility: hidden;}
    .block-container {position: relative; z-index: 1; max-width: 1450px;}

    .topbar {
      border: 1px solid rgba(113, 143, 183, 0.22);
      border-radius: 14px;
      background: rgba(3, 12, 29, 0.88);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.8rem 1rem;
      margin-bottom: 1rem;
    }
    .brand {display:flex; align-items:center; gap:0.8rem;}
    .brand-icon {width:40px;height:40px;border-radius:10px;border:1px solid rgba(0, 217, 232, 0.6);display:flex;align-items:center;justify-content:center;color:var(--cyan);background:rgba(0,217,232,0.08)}
    .brand h1 {margin:0;font-family:"Orbitron",sans-serif;font-size:1.1rem;letter-spacing:1px;color:#f3f8ff;}
    .brand p {margin:0;color:var(--muted);font-size:0.78rem;letter-spacing:1.6px;}

    .pill {border:1px solid rgba(104,134,172,0.28);border-radius:999px;padding:0.3rem 0.65rem;color:var(--muted);background:rgba(4,11,24,0.72);font-size:0.74rem;letter-spacing:1px;font-family:"Orbitron",sans-serif;display:inline-block;margin-right:0.35rem;}
    .pill.active {color:var(--lime);border-color:rgba(156,255,58,0.4);}

    .hero {border:1px solid rgba(113,143,183,0.2);border-radius:26px;background:linear-gradient(180deg, rgba(4,14,33,0.92), rgba(3,11,26,0.9));padding:1.1rem 1.25rem;margin-bottom:1rem;}
    .hero-title {margin:0;font-size:2.2rem;font-family:"Orbitron",sans-serif;color:#edf4ff;}
    .hero-title span {color:var(--cyan);}
    .sync {font-family:"Orbitron",sans-serif;font-size:0.75rem;letter-spacing:3px;color:var(--lime);margin-bottom:0.35rem;}
    .section-title {font-family:"Orbitron",sans-serif;color:#d9e8ff;letter-spacing:1.4px;margin-top:0.2rem;margin-bottom:0.4rem;}
    .panel-box {border:1px solid rgba(89,120,156,0.28);border-radius:16px;padding:0.8rem;background:linear-gradient(180deg, rgba(6,18,40,0.72), rgba(3,12,28,0.63));margin-bottom:0.8rem;}

    div[data-testid="stMetric"] {border-radius:18px;border:1.4px solid rgba(120,160,205,0.3);background:linear-gradient(180deg, rgba(4,13,31,0.85), rgba(3,10,24,0.82));padding:0.9rem;min-height:120px;}
    div[data-testid="stMetricLabel"] p {color:#91a7c8 !important;font-family:"Orbitron",sans-serif;letter-spacing:1.8px;font-size:0.74rem !important;text-transform:uppercase;}
    div[data-testid="stMetricValue"] {font-family:"Orbitron",sans-serif;font-size:2rem;color:#eff6ff !important;}

    div[data-testid="stDataFrame"] {border:1px solid rgba(89,120,156,0.3);border-radius:12px;background:rgba(2,10,25,0.8);}
    .stTabs [role="tab"] {font-family:"Orbitron",sans-serif;letter-spacing:1px;}

    label, .stMarkdown p, .stText, .stSelectbox label, .stMultiSelect label {color: #d6e4fb !important;}
    button[kind="secondary"], button[kind="primary"] {border-radius:12px !important;border:1px solid rgba(0,217,232,0.45) !important;background:rgba(2,17,34,0.88) !important;color:var(--cyan) !important;font-family:"Orbitron",sans-serif !important;letter-spacing:1.1px !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


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

    st.markdown("### <span class='section-title'>ACCESS CONTROL</span>", unsafe_allow_html=True)
    with st.container(border=True):
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", use_container_width=True):
            entry = users.get(username)
            if entry and password == entry.get("password"):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["role"] = entry.get("role", "viewer")
                st.rerun()
            else:
                st.error("Invalid username or password")

    st.info("Default demo credentials: admin/admin123 or viewer/viewer123")
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
    st.markdown(f"### <span class='section-title'>{title}</span>", unsafe_allow_html=True)
    st.dataframe(reports[report_key], use_container_width=True, height=320)
    st.download_button(
        f"Download {title} (CSV)",
        data=to_csv_download(reports[report_key]),
        file_name=filename,
        mime="text/csv",
    )


st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="brand-icon">✦</div>
        <div>
          <h1>WATU INTELLIGENCE 2026</h1>
          <p>SALES COMMAND GRID</p>
        </div>
      </div>
      <div>
        <span class="pill">[ 0X-WATU-AUDIT ]</span>
        <span class="pill active">ACTIVE_NODE</span>
      </div>
    </div>
    <div class="hero">
      <div class="sync">SYNCHRONIZED_ACTIVE</div>
      <h2 class="hero-title">WATU DAILY <span>SALES</span></h2>
    </div>
    """,
    unsafe_allow_html=True,
)

role = login_ui()
if role is None:
    st.stop()

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

st.markdown("### <span class='section-title'>UPLINK DATA</span>", unsafe_allow_html=True)
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
st.markdown("### <span class='section-title'>DATA QUALITY</span>", unsafe_allow_html=True)
st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
q1, q2, q3, q4 = st.columns(4)
q1.metric("Total Rows", quality.total_rows)
q2.metric("Mobile Device Rows", quality.mobile_rows)
q3.metric("Excluded Non-Mobile", quality.excluded_non_mobile)
q4.metric("Duplicate Rows", quality.duplicate_rows)
q5, q6, q7 = st.columns(3)
q5.metric("Invalid Dates", quality.invalid_dates)
q6.metric("Missing Dealership", quality.missing_dealership)
q7.metric("Missing Branch/Model", quality.missing_branch + quality.missing_model)
st.markdown("</div>", unsafe_allow_html=True)

with st.status("Preparing analytics...", expanded=False) as status:
    status.write("Applying core rules")
    prepared, _ = process_dataframe(df)
    status.update(label="Analytics ready", state="complete")

if prepared.empty:
    st.warning("No rows found where type is 'Mobile device'.")
    st.stop()

st.markdown("### <span class='section-title'>FILTER MATRIX</span>", unsafe_allow_html=True)
st.markdown("<div class='panel-box'>", unsafe_allow_html=True)

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
st.markdown("</div>", unsafe_allow_html=True)

if filtered.empty:
    st.warning("No records match selected filters.")
    st.stop()

reports = build_reports(filtered)
comparison = compare_periods(filtered)

st.markdown("### <span class='section-title'>SYSTEM SNAPSHOT</span>", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("TOTAL UNITS", int(len(filtered)))
m2.metric("DEALERSHIPS", int(reports["sales_by_dealership"].shape[0]))
m3.metric("BRANCHES", int(reports["sales_by_branch"].shape[0]))
m4.metric("MODELS", int(reports["sales_by_model"].shape[0]))
m5.metric("PERIOD CHANGE", f"{comparison['pct_change']:.1f}%")

st.markdown("### <span class='section-title'>TREND VISUALS</span>", unsafe_allow_html=True)
st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
ch1, ch2 = st.columns(2)
with ch1:
    if "sales_by_week" in reports:
        week_df = reports["sales_by_week"].rename(columns={"week_start": "period"}).set_index("period")
        st.line_chart(week_df["sales_count"], color="#00d9e8")
    else:
        st.info("Weekly trend unavailable (no valid dates).")
with ch2:
    branch_top = reports["sales_by_branch"].head(10).set_index("branch_office")
    st.bar_chart(branch_top["sales_count"], color="#9cff3a")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### <span class='section-title'>EXPORT MATRIX</span>", unsafe_allow_html=True)
st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
excel_bytes = to_excel_download(reports)
pdf_bytes = to_pdf_download(reports, comparison, int(len(filtered)))

ex1, ex2, ex3 = st.columns(3)
ex1.download_button(
    "EXPORT FULL REPORT (EXCEL)",
    data=excel_bytes,
    file_name="management_sales_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
ex2.download_button(
    "EXPORT FILTERED RAW DATA (CSV)",
    data=to_csv_download(filtered),
    file_name="filtered_mobile_device_data.csv",
    mime="text/csv",
)
ex3.download_button(
    "EXPORT BRIEFING (PDF)",
    data=pdf_bytes,
    file_name="management_briefing.pdf",
    mime="application/pdf",
)

if role == "admin":
    st.markdown("#### Email Report")
    recipients_text = st.text_input("Recipients (comma-separated)", value="")
    email_subject = st.text_input("Subject", value=f"Management Sales Report - {datetime.now().strftime('%Y-%m-%d')}")

    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("SMTP_SENDER", smtp_user)

    if st.button("SEND REPORT EMAIL"):
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
            st.error(msg + " | Configure SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASSWORD/SMTP_SENDER")
else:
    st.caption("Email dispatch is admin-only.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### <span class='section-title'>MARKET CONCENTRATION</span>", unsafe_allow_html=True)
st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
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
st.markdown("</div>", unsafe_allow_html=True)

if "sales_by_month" in reports or "sales_by_week" in reports:
    st.markdown("### <span class='section-title'>TIME TRENDS</span>", unsafe_allow_html=True)
    st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    if "sales_by_month" in reports:
        with t1:
            show_report_section("Sales by Month", "sales_by_month", reports, "sales_by_month.csv")
    if "sales_by_week" in reports:
        with t2:
            show_report_section("Sales by Week", "sales_by_week", reports, "sales_by_week.csv")
    st.markdown("</div>", unsafe_allow_html=True)
