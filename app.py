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

st.set_page_config(page_title="Management Sales Report", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
      --bg: #f4f7fc;
      --surface: #ffffff;
      --line: #e3e8f3;
      --text: #111827;
      --muted: #667085;
      --brand: #1f5fd1;
      --brand-2: #4c8dff;
      --ink: #0b1220;
    }

    .stApp {
      background:
        radial-gradient(1400px 360px at 12% -12%, rgba(76, 141, 255, 0.16), transparent 60%),
        radial-gradient(1000px 280px at 88% -12%, rgba(31, 95, 209, 0.12), transparent 58%),
        linear-gradient(180deg, #f8faff 0%, var(--bg) 58%);
      color: var(--text);
      font-family: "Plus Jakarta Sans", sans-serif;
    }

    .block-container {
      max-width: 1400px;
      padding-top: 1rem;
      padding-bottom: 2.5rem;
    }

    h1, h2, h3, h4, h5, h6, p, label {
      font-family: "Plus Jakarta Sans", sans-serif !important;
    }

    .hero {
      background: linear-gradient(120deg, #123a85 0%, #1f5fd1 58%, #4c8dff 100%);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 18px;
      padding: 1.15rem 1.35rem;
      color: #ffffff;
      box-shadow: 0 16px 34px rgba(19, 58, 133, 0.28);
      margin-bottom: 1rem;
    }

    .hero-title {
      margin: 0;
      font-size: 1.55rem;
      font-weight: 800;
      letter-spacing: -0.2px;
    }

    .hero-sub {
      margin: 0.35rem 0 0;
      color: #dce9ff;
      font-size: 0.96rem;
      font-weight: 500;
    }

    .badge-row {
      display: flex;
      gap: 0.5rem;
      margin-top: 0.85rem;
      flex-wrap: wrap;
    }

    .hero-badge {
      background: rgba(255, 255, 255, 0.14);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 999px;
      color: #e9f1ff;
      font-size: 0.76rem;
      padding: 0.22rem 0.62rem;
      font-weight: 600;
      letter-spacing: 0.2px;
    }

    .section-title {
      margin-top: 0.1rem;
      margin-bottom: 0.2rem;
      font-size: 1.08rem;
      font-weight: 700;
      color: var(--ink);
      letter-spacing: -0.1px;
    }

    .section-sub {
      margin-top: 0;
      margin-bottom: 0.6rem;
      color: var(--muted);
      font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 0.85rem;
      box-shadow: 0 5px 16px rgba(17, 24, 39, 0.045);
    }

    div[data-testid="stMetricLabel"] p {
      color: #64748b !important;
      text-transform: uppercase;
      letter-spacing: 0.55px;
      font-size: 0.76rem;
      font-weight: 700;
    }

    div[data-testid="stMetricValue"] {
      color: #0f172a !important;
      font-weight: 800;
      font-size: 1.8rem;
    }

    div[data-testid="stDataFrame"] {
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 12px rgba(17, 24, 39, 0.04);
    }

    div[data-testid="stSidebar"] {
      border-right: 1px solid #e8edf7;
    }

    button[kind="primary"] {
      background: var(--brand) !important;
      border-color: var(--brand) !important;
    }

    button[kind="secondary"] {
      border-color: #ccd6ea !important;
    }
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


def section_header(title: str, subtitle: str) -> None:
    st.markdown(f"<p class='section-title'>{title}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='section-sub'>{subtitle}</p>", unsafe_allow_html=True)


def login_ui() -> Optional[str]:
    users = get_auth_config()
    if st.session_state.get("authenticated"):
        return st.session_state.get("role")

    left, center, right = st.columns([1, 1.1, 1])
    with center:
        st.markdown("## Management Sales Report")
        st.caption("Sign in to continue")
        with st.form("login_form", border=True):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            entry = users.get(username)
            if entry and password == entry.get("password"):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["role"] = entry.get("role", "viewer")
                st.rerun()
            st.error("Invalid username or password")

        st.caption("Demo credentials: admin/admin123 or viewer/viewer123")
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
        use_container_width=True,
    )


role = login_ui()
if role is None:
    st.stop()

st.markdown(
    """
    <div class="hero">
      <p class="hero-title">Management Sales Dashboard</p>
      <p class="hero-sub">Real-time sales intelligence by dealership, model, branch, and period.</p>
      <div class="badge-row">
        <span class="hero-badge">CSV Uploader</span>
        <span class="hero-badge">Data Quality Checks</span>
        <span class="hero-badge">Excel + PDF Exports</span>
        <span class="hero-badge">Role-Based Access</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Session")
    st.write(f"User: `{st.session_state.get('username')}`")
    st.write(f"Role: `{role}`")
    if st.button("Logout", use_container_width=True):
        for k in ["authenticated", "username", "role", "uploaded_source"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.header("Upload History")
    history = load_upload_history()
    options = [f"{h['timestamp']} | {h['original_name']} ({h['rows']} rows)" for h in history]
    selected_idx = st.selectbox("Reopen previous upload", options=(['None'] + options), index=0)
    if selected_idx != "None":
        selected = history[options.index(selected_idx)]
        if st.button("Load Selected", use_container_width=True):
            st.session_state["uploaded_source"] = {
                "saved_name": selected["saved_name"],
                "name": selected["original_name"],
            }
            st.rerun()

with st.container(border=True):
    section_header("Data Source", "Upload your latest CSV sales file.")
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
        st.session_state["uploaded_source"] = {
            "saved_name": None,
            "name": uploaded_file.name,
            "bytes": file_bytes,
        }
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
        "Expected headers: Client Mobile No, Branch Office, Dealership, disbursedon_date, registration_no, chasis_no, make, model, type"
    )
    st.stop()

quality = data_quality_summary(df)
with st.container(border=True):
    section_header("Data Quality", "Quick validation summary of uploaded records.")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Total Rows", quality.total_rows)
    q2.metric("Mobile Rows", quality.mobile_rows)
    q3.metric("Excluded", quality.excluded_non_mobile)
    q4.metric("Duplicates", quality.duplicate_rows)
    q5, q6, q7 = st.columns(3)
    q5.metric("Invalid Dates", quality.invalid_dates)
    q6.metric("Missing Dealership", quality.missing_dealership)
    q7.metric("Missing Branch/Model", quality.missing_branch + quality.missing_model)

with st.status("Preparing analytics...", expanded=False) as status:
    status.write("Applying sales rules")
    prepared, _ = process_dataframe(df)
    status.update(label="Analytics ready", state="complete")

if prepared.empty:
    st.warning("No rows found where type is 'Mobile device'.")
    st.stop()

with st.container(border=True):
    section_header("Filters", "Refine data before generating reports and exports.")
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
            selected_range = st.date_input(
                "Custom Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
            )
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

with st.container(border=True):
    section_header("Snapshot", "High-level KPIs for current filtered period.")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Units", int(len(filtered)))
    m2.metric("Dealerships", int(reports["sales_by_dealership"].shape[0]))
    m3.metric("Branches", int(reports["sales_by_branch"].shape[0]))
    m4.metric("Models", int(reports["sales_by_model"].shape[0]))
    m5.metric("Period Change", f"{comparison['pct_change']:.1f}%")

with st.container(border=True):
    section_header("Trends", "Weekly momentum and top branch contribution.")
    c1, c2 = st.columns(2)
    with c1:
        if "sales_by_week" in reports:
            week_df = reports["sales_by_week"].rename(columns={"week_start": "period"}).set_index("period")
            st.line_chart(week_df["sales_count"])
        else:
            st.info("Weekly trend unavailable (no valid dates).")
    with c2:
        branch_top = reports["sales_by_branch"].head(10).set_index("branch_office")
        st.bar_chart(branch_top["sales_count"])

with st.container(border=True):
    section_header("Exports", "Download full reports and briefing files.")
    excel_bytes = to_excel_download(reports)
    pdf_bytes = to_pdf_download(reports, comparison, int(len(filtered)))

    ex1, ex2, ex3 = st.columns(3)
    ex1.download_button(
        "Export Full Report (Excel)",
        data=excel_bytes,
        file_name="management_sales_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    ex2.download_button(
        "Export Filtered Raw Data (CSV)",
        data=to_csv_download(filtered),
        file_name="filtered_mobile_device_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
    ex3.download_button(
        "Export Briefing (PDF)",
        data=pdf_bytes,
        file_name="management_briefing.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

if role == "admin":
    with st.container(border=True):
        section_header("Email Report (Admin)", "Send the latest Excel report directly to stakeholders.")
        recipients_text = st.text_input("Recipients (comma-separated)", value="")
        email_subject = st.text_input(
            "Subject", value=f"Management Sales Report - {datetime.now().strftime('%Y-%m-%d')}"
        )

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

with st.container(border=True):
    section_header("Reports", "Detailed sales views by dealership, model, and branch.")
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
    with st.container(border=True):
        section_header("Time Reports", "Month-over-month and week-over-week sales summaries.")
        t1, t2 = st.columns(2)
        if "sales_by_month" in reports:
            with t1:
                show_report_section("Sales by Month", "sales_by_month", reports, "sales_by_month.csv")
        if "sales_by_week" in reports:
            with t2:
                show_report_section("Sales by Week", "sales_by_week", reports, "sales_by_week.csv")
