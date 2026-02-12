import os
from datetime import datetime
from io import BytesIO
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from reporting import (
    apply_date_filter,
    auto_insights,
    build_reports,
    cleaning_assistant,
    compare_periods,
    data_dictionary,
    data_quality_summary,
    delete_filter_preset,
    detect_anomalies,
    filter_data,
    forecast_sales,
    generate_narrative,
    load_audit_log,
    load_filter_presets,
    load_snapshot_df,
    load_upload_history,
    log_audit_event,
    map_columns,
    nl_qa,
    prepare_data,
    recommend_targets,
    root_cause_suggestions,
    save_filter_preset,
    save_upload_snapshot,
    schedule_report_helper,
    send_email_report,
    smart_alerts,
    target_vs_actual,
    to_csv_download,
    to_excel_download,
    to_pdf_download,
    validate_required_columns,
    what_if_simulation,
)

st.set_page_config(page_title="Management Sales Report", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp {font-family: "Inter", sans-serif; background: #eff1f3;}
    .block-container {max-width: 1500px; padding-top: 0.45rem; padding-bottom: 1rem;}
    .topbar {background: #fff; border: 1px solid #d8dde3; border-radius: 10px; padding: 0.55rem 0.8rem; margin-bottom: 0.5rem;}
    .topbar h1 {margin:0; font-size: 0.98rem; font-weight:700; color:#1c2733;}
    .topbar p {margin:0.12rem 0 0; color:#6e7782; font-size:0.78rem;}
    .chip {display:inline-block; border:1px solid #d8dde3; border-radius:999px; padding:0.18rem 0.54rem; font-size:0.7rem; color:#3d4854; background:#f7f9fb; font-weight:600; margin-right:0.3rem;}
    .chip.active {background:#d7ece8; border-color:#b5d7d1; color:#0b6458;}
    .section-header {margin: 0; font-size: 0.92rem; font-weight: 700; color: #212a36;}
    .section-note {margin: 0.1rem 0 0.35rem; color: #6e7782; font-size: 0.76rem;}
    div[data-testid="stMetric"] {background:#fff; border:1px solid #d8dde3; border-radius:9px; padding:0.48rem;}
    div[data-testid="stMetricLabel"] p {font-size:0.66rem; text-transform:uppercase; letter-spacing:0.4px;}
    div[data-testid="stMetricValue"] {font-size:1.38rem; font-weight:700;}
    div[data-testid="stDataFrame"] {border:1px solid #d8dde3; border-radius:8px; overflow:hidden; background:#fff;}
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
    st.markdown(f"<p class='section-header'>{title}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='section-note'>{subtitle}</p>", unsafe_allow_html=True)


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
    st.dataframe(reports[report_key], use_container_width=True, height=290)
    st.download_button(
        f"Download {title} (CSV)",
        data=to_csv_download(reports[report_key]),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
        key=f"dl_{report_key}",
    )


role = login_ui()
if role is None:
    st.stop()

username = st.session_state.get("username", "unknown")

st.markdown(
    """
    <div class="topbar">
      <h1>Sales Performance Workspace</h1>
      <p>Enterprise view for dealership, branch, and model performance with AI assistance.</p>
      <div>
        <span class="chip active">Status</span>
        <span class="chip">Sales</span>
        <span class="chip">Forecast</span>
        <span class="chip">AI Insights</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Navigation")
    st.markdown("Dashboard\n\nAI Tools\n\nExports\n\nAudit")
    st.header("Session")
    st.write(f"User: `{username}`")
    st.write(f"Role: `{role}`")
    performance_mode = st.toggle("Performance Mode", value=True, help="Skip heavy sections for faster rendering")
    if st.button("Logout", use_container_width=True):
        for k in ["authenticated", "username", "role", "uploaded_source", "preset_payload"]:
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
            log_audit_event(username, "load_history", {"file": selected["original_name"]})
            st.rerun()

with st.expander("Data Dictionary & Validation Rules", expanded=False):
    st.dataframe(data_dictionary(), use_container_width=True)

with st.container(border=True):
    section_header("Data Source", "Upload latest CSV extract")
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
    log_audit_event(username, "upload", {"file": source_name, "rows": int(raw_df.shape[0])})
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
    section_header("Key Sales Figures", "Summary of quality and inclusion checks")
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

# Filter preset management
presets = load_filter_presets()
user_presets = [p for p in presets if p.get("user") == username]

with st.container(border=True):
    section_header("Filters", "Choose period and business dimensions")

    preset_options = ["None"] + [f"{p['name']} ({p['created_at']})" for p in user_presets]
    preset_choice = st.selectbox("Saved Presets", preset_options, index=0)

    preset_payload = None
    if preset_choice != "None":
        preset_payload = user_presets[preset_options.index(preset_choice) - 1]["payload"]

    has_valid_dates = prepared["disbursed_date"].notna().any()
    col1, col2, col3 = st.columns(3)

    default_mode = preset_payload.get("filter_mode", "All Dates") if preset_payload else "All Dates"
    filter_mode = col1.selectbox("Report Period", ["All Dates", "This Week", "This Month", "Custom Range"], index=["All Dates", "This Week", "This Month", "Custom Range"].index(default_mode))

    all_dealers = sorted([x for x in prepared["dealership"].dropna().unique().tolist() if str(x).strip()])
    all_branches = sorted([x for x in prepared["branch_office"].dropna().unique().tolist() if str(x).strip()])
    all_models = sorted([x for x in prepared["model"].dropna().unique().tolist() if str(x).strip()])
    all_makes = sorted([x for x in prepared["make"].dropna().unique().tolist() if str(x).strip()])

    sel_dealers = col2.multiselect("Dealership", options=all_dealers, default=(preset_payload.get("dealers", []) if preset_payload else []))
    sel_branches = col3.multiselect("Branch", options=all_branches, default=(preset_payload.get("branches", []) if preset_payload else []))

    col4, col5 = st.columns(2)
    sel_models = col4.multiselect("Model", options=all_models, default=(preset_payload.get("models", []) if preset_payload else []))
    sel_makes = col5.multiselect("Make", options=all_makes, default=(preset_payload.get("makes", []) if preset_payload else []))

    start_dt = None
    end_dt = None
    if filter_mode == "Custom Range":
        if has_valid_dates:
            min_date = prepared["disbursed_date"].min().date()
            max_date = prepared["disbursed_date"].max().date()
            default_range = (min_date, max_date)
            if preset_payload and preset_payload.get("start_dt") and preset_payload.get("end_dt"):
                default_range = (
                    pd.to_datetime(preset_payload["start_dt"]).date(),
                    pd.to_datetime(preset_payload["end_dt"]).date(),
                )
            selected_range = st.date_input("Custom Range", value=default_range, min_value=min_date, max_value=max_date)
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

    p1, p2 = st.columns([2, 1])
    preset_name = p1.text_input("Preset name", value="")
    if p1.button("Save Current Preset"):
        if not preset_name.strip():
            st.warning("Enter preset name")
        else:
            payload = {
                "filter_mode": filter_mode,
                "dealers": sel_dealers,
                "branches": sel_branches,
                "models": sel_models,
                "makes": sel_makes,
                "start_dt": str(start_dt) if start_dt else "",
                "end_dt": str(end_dt) if end_dt else "",
            }
            save_filter_preset(preset_name.strip(), username, payload)
            log_audit_event(username, "save_preset", {"name": preset_name.strip()})
            st.success("Preset saved")
            st.rerun()

    if preset_choice != "None" and p2.button("Delete Selected Preset"):
        preset_id = user_presets[preset_options.index(preset_choice) - 1]["id"]
        delete_filter_preset(preset_id)
        log_audit_event(username, "delete_preset", {"id": preset_id})
        st.success("Preset deleted")
        st.rerun()

filtered = filter_data(prepared, sel_dealers, sel_branches, sel_models, sel_makes)
filtered = apply_date_filter(filtered, filter_mode, start_dt, end_dt)

if filtered.empty:
    st.warning("No records match selected filters.")
    st.stop()

# Interactive drill-through
with st.container(border=True):
    section_header("Interactive Drill-Through", "Use top performers to drill into report context")
    tmp_reports = build_reports(filtered)
    top_dealers = ["All"] + tmp_reports["sales_by_dealership"]["dealership"].head(10).astype(str).tolist()
    top_branches = ["All"] + tmp_reports["sales_by_branch"]["branch_office"].head(10).astype(str).tolist()
    d1, d2 = st.columns(2)
    drill_dealer = d1.selectbox("Focus Dealership", top_dealers)
    drill_branch = d2.selectbox("Focus Branch", top_branches)

analysis_df = filtered.copy()
if drill_dealer != "All":
    analysis_df = analysis_df[analysis_df["dealership"] == drill_dealer]
if drill_branch != "All":
    analysis_df = analysis_df[analysis_df["branch_office"] == drill_branch]

if analysis_df.empty:
    st.warning("Drill-through selection returned no rows. Reverting to filtered data.")
    analysis_df = filtered

reports = build_reports(analysis_df)
comparison = compare_periods(analysis_df)
insights = auto_insights(reports, comparison, quality)
alerts = smart_alerts(reports, quality, comparison)
root_causes = root_cause_suggestions(analysis_df)
anomalies = detect_anomalies(analysis_df)

with st.container(border=True):
    section_header("Executive Summary", "Top facts and AI-generated highlights")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Units", int(len(analysis_df)))
    m2.metric("Dealerships", int(reports["sales_by_dealership"].shape[0]))
    m3.metric("Branches", int(reports["sales_by_branch"].shape[0]))
    m4.metric("Models", int(reports["sales_by_model"].shape[0]))
    m5.metric("Period Change", f"{comparison['pct_change']:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Auto Insights**")
        for ins in insights:
            st.write(f"- {ins}")
    with c2:
        st.markdown("**Root Cause Suggestions**")
        for rc in root_causes:
            st.write(f"- {rc}")

with st.container(border=True):
    section_header("Smart Alerts", "Automated thresholds and risk notifications")
    if alerts:
        for a in alerts:
            if a["level"] == "high":
                st.error(a["message"])
            elif a["level"] == "medium":
                st.warning(a["message"])
            else:
                st.info(a["message"])
    else:
        st.success("No alerts triggered in current view.")

with st.container(border=True):
    section_header("Anomaly Detection", "Outlier weeks identified from time-series z-score")
    if anomalies.empty:
        st.info("No significant anomalies detected.")
    else:
        st.dataframe(anomalies, use_container_width=True)

with st.container(border=True):
    section_header("Forecasting", "Next period predictions by dealership/model")
    fc_col1, fc_col2 = st.columns(2)
    with fc_col1:
        fc_dealer = forecast_sales(analysis_df, by="dealership", periods=1)
        st.markdown("**Dealership Forecast (Next Month)**")
        st.dataframe(fc_dealer.head(15), use_container_width=True)
    with fc_col2:
        fc_model = forecast_sales(analysis_df, by="model", periods=1)
        st.markdown("**Model Forecast (Next Month)**")
        st.dataframe(fc_model.head(15), use_container_width=True)

with st.container(border=True):
    section_header("Target vs Actual", "Upload target file to monitor achievement")
    st.caption("Required target columns: dealership, branch_office, target_sales")
    target_file = st.file_uploader("Upload Targets CSV", type=["csv"], key="targets_upload")
    if target_file is not None:
        targets_df = pd.read_csv(target_file, dtype=str)
        tva = target_vs_actual(analysis_df, targets_df)
        if tva.empty:
            st.warning("Invalid target file format.")
        else:
            st.dataframe(tva, use_container_width=True)
    rec_targets = recommend_targets(analysis_df)
    st.markdown("**AI Target Recommendations**")
    st.dataframe(rec_targets.head(15), use_container_width=True)

with st.container(border=True):
    section_header("Natural Language Q&A", "Ask questions about current filtered data")
    q = st.text_input("Ask a question", placeholder="Which branch has highest sales?")
    if st.button("Answer"):
        answer = nl_qa(q, analysis_df, reports)
        st.info(answer)
        log_audit_event(username, "qa_query", {"question": q, "answer": answer})

with st.container(border=True):
    section_header("Data Cleaning Assistant", "Detected cleaning opportunities and normalization hints")
    cleaning = cleaning_assistant(df)
    if not cleaning["summary"].empty:
        st.dataframe(cleaning["summary"], use_container_width=True)
    else:
        st.success("No major cleaning issues detected.")
    if not cleaning["dealership_suggestions"].empty:
        with st.expander("Dealership Normalization Suggestions"):
            st.dataframe(cleaning["dealership_suggestions"], use_container_width=True)

with st.container(border=True):
    section_header("Narrative Report Generator", "One-click executive text briefing")
    narrative = generate_narrative(reports, comparison, alerts, insights)
    st.text_area("Narrative", value=narrative, height=180)
    st.download_button(
        "Download Narrative TXT",
        data=narrative.encode("utf-8"),
        file_name="management_narrative.txt",
        mime="text/plain",
        use_container_width=True,
    )

with st.container(border=True):
    section_header("What-if Simulator", "Estimate impact of strategic uplift scenarios")
    w1, w2 = st.columns(2)
    dealer_uplift = w1.slider("Dealership uplift %", min_value=-30, max_value=50, value=10)
    branch_uplift = w2.slider("Branch uplift %", min_value=-30, max_value=50, value=5)
    sim = what_if_simulation(int(len(analysis_df)), float(dealer_uplift), float(branch_uplift))
    s1, s2, s3 = st.columns(3)
    s1.metric("Base Total", int(sim["base_total"]))
    s2.metric("Projected Total", int(sim["projected_total"]))
    s3.metric("Delta", int(sim["delta"]))

with st.container(border=True):
    section_header("Scheduled Report Helper", "Generate a deployment-ready schedule plan")
    sch1, sch2, sch3 = st.columns(3)
    freq = sch1.selectbox("Frequency", ["Daily", "Weekly"])
    day = sch2.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    time_24h = sch3.text_input("Time (24h)", value="09:00")
    sched_recipients = st.text_input("Recipients for scheduled run", value="")
    if st.button("Generate Schedule Plan"):
        plan = schedule_report_helper(freq, day, time_24h, [x.strip() for x in sched_recipients.split(",") if x.strip()])
        st.success(plan)

with st.container(border=True):
    section_header("Forecast and Trends", "Operational trend indicators")
    c1, c2 = st.columns(2)
    with c1:
        if "sales_by_week" in reports:
            week_df = reports["sales_by_week"].rename(columns={"week_start": "period"}).set_index("period")
            st.line_chart(week_df["sales_count"], color="#0f7b6c")
        else:
            st.info("Weekly trend unavailable (no valid dates).")
    with c2:
        branch_top = reports["sales_by_branch"].head(10).set_index("branch_office")
        st.bar_chart(branch_top["sales_count"], color="#2a9d8f")

with st.container(border=True):
    section_header("Exports", "Distribute validated reporting outputs")
    excel_bytes = to_excel_download(reports)
    pdf_bytes = to_pdf_download(reports, comparison, int(len(analysis_df)))

    if role == "viewer":
        st.caption("Viewer role: CSV exports only")
        st.download_button(
            "Export Filtered Raw Data (CSV)",
            data=to_csv_download(analysis_df),
            file_name="filtered_mobile_device_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
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
            data=to_csv_download(analysis_df),
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
        section_header("Email Report (Admin)", "Send current report pack to stakeholders")
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
                log_audit_event(username, "send_email", {"recipients": recipients, "subject": email_subject})
            else:
                st.error(msg + " | Configure SMTP_* env vars")

with st.container(border=True):
    section_header("Detailed Reports", "Data-heavy breakdowns for operational review")
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
        section_header("Time Reports", "Month and week aggregated outputs")
        t1, t2 = st.columns(2)
        if "sales_by_month" in reports:
            with t1:
                show_report_section("Sales by Month", "sales_by_month", reports, "sales_by_month.csv")
        if "sales_by_week" in reports:
            with t2:
                show_report_section("Sales by Week", "sales_by_week", reports, "sales_by_week.csv")

if not performance_mode:
    with st.container(border=True):
        section_header("Audit Trail", "Recent user and system actions")
        audit = pd.DataFrame(load_audit_log())
        if audit.empty:
            st.info("No audit entries yet.")
        else:
            st.dataframe(audit.head(200), use_container_width=True)
