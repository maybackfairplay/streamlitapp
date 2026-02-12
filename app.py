import re
from io import BytesIO, StringIO
from typing import Dict, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="WATU Daily Sales", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Rajdhani:wght@500;700&display=swap');

    :root {
      --bg0: #030916;
      --bg1: #061127;
      --panel: rgba(4, 14, 33, 0.84);
      --panel-solid: #081428;
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
    }

    #MainMenu, header[data-testid="stHeader"], footer {visibility: hidden;}
    [data-testid="stAppViewContainer"] {padding-top: 0.5rem;}

    .topbar {
      border: 1px solid rgba(113, 143, 183, 0.22);
      border-radius: 14px;
      background: rgba(3, 12, 29, 0.88);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.8rem 1rem;
      margin-bottom: 1rem;
      box-shadow: 0 0 0 1px rgba(16, 33, 58, 0.5), 0 20px 40px rgba(1, 4, 12, 0.5);
    }

    .brand {display: flex; align-items: center; gap: 0.8rem;}
    .brand-icon {
      width: 40px; height: 40px; border-radius: 10px;
      border: 1px solid rgba(0, 217, 232, 0.6);
      display:flex; align-items:center; justify-content:center;
      color: var(--cyan); font-size: 1.2rem;
      background: rgba(0, 217, 232, 0.08);
      box-shadow: 0 0 20px rgba(0, 217, 232, 0.25);
    }
    .brand h1 {
      margin: 0; font-family: "Orbitron", sans-serif;
      font-size: 1.1rem; letter-spacing: 1px; color: #f3f8ff;
    }
    .brand p {margin: 0; color: var(--muted); font-size: 0.78rem; letter-spacing: 1.6px;}

    .status-wrap {display:flex; gap:0.6rem; align-items:center;}
    .pill {
      border: 1px solid rgba(104, 134, 172, 0.28);
      border-radius: 999px;
      padding: 0.35rem 0.7rem;
      color: var(--muted);
      background: rgba(4, 11, 24, 0.72);
      font-size: 0.78rem;
      letter-spacing: 1.1px;
      font-family: "Orbitron", sans-serif;
    }
    .pill.active {color: var(--lime); border-color: rgba(156, 255, 58, 0.4);}

    .hero {
      border: 1px solid rgba(113, 143, 183, 0.2);
      border-radius: 26px;
      background: linear-gradient(180deg, rgba(4, 14, 33, 0.92), rgba(3, 11, 26, 0.9));
      padding: 1.3rem 1.4rem;
      margin-bottom: 1.2rem;
      box-shadow: inset 0 1px 0 rgba(129, 160, 198, 0.2);
    }
    .sync {
      font-family: "Orbitron", sans-serif;
      font-size: 0.78rem;
      letter-spacing: 3.2px;
      color: var(--lime);
      margin-bottom: 0.35rem;
    }
    .hero-title {
      margin: 0;
      font-size: 2.25rem;
      font-family: "Orbitron", sans-serif;
      line-height: 1.05;
      color: #edf4ff;
    }
    .hero-title span {color: var(--cyan);}

    .cmd-row {
      display:flex; gap:0.7rem; flex-wrap:wrap; margin-top: 0.9rem;
    }
    .cmd {
      border: 1px solid rgba(105, 137, 175, 0.28);
      border-radius: 14px;
      padding: 0.45rem 0.9rem;
      font-family: "Orbitron", sans-serif;
      font-size: 0.8rem;
      letter-spacing: 1.1px;
      color: #d4e1f6;
      background: rgba(5, 12, 26, 0.75);
    }
    .cmd.highlight {
      color: var(--cyan);
      border-color: rgba(0, 217, 232, 0.55);
      box-shadow: 0 0 18px rgba(0, 217, 232, 0.18);
    }

    .section-title {
      font-family: "Orbitron", sans-serif;
      color: #d9e8ff;
      letter-spacing: 1.4px;
      margin-top: 0.2rem;
      margin-bottom: 0.5rem;
    }

    div[data-testid="stMetric"] {
      border-radius: 22px;
      border: 1.6px solid rgba(120, 160, 205, 0.3);
      background: linear-gradient(180deg, rgba(4, 13, 31, 0.85), rgba(3, 10, 24, 0.82));
      padding: 1rem;
      min-height: 130px;
      box-shadow: inset 0 0 0 1px rgba(17, 34, 59, 0.5);
    }
    div[data-testid="stMetricLabel"] p {
      color: #91a7c8 !important;
      font-family: "Orbitron", sans-serif;
      letter-spacing: 2px;
      font-size: 0.78rem !important;
      text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
      font-family: "Orbitron", sans-serif;
      font-size: 2.2rem;
      color: #eff6ff !important;
    }

    .border-cyan div[data-testid="stMetric"] {border-color: rgba(0, 217, 232, 0.85); box-shadow: 0 0 25px rgba(0, 217, 232, 0.2);}    
    .border-lime div[data-testid="stMetric"] {border-color: rgba(156, 255, 58, 0.8); box-shadow: 0 0 24px rgba(156, 255, 58, 0.14);}    
    .border-pink div[data-testid="stMetric"] {border-color: rgba(255, 46, 115, 0.8); box-shadow: 0 0 24px rgba(255, 46, 115, 0.16);}    
    .border-amber div[data-testid="stMetric"] {border-color: rgba(255, 191, 61, 0.8); box-shadow: 0 0 24px rgba(255, 191, 61, 0.18);}    

    div[data-testid="stFileUploader"],
    div[data-testid="stSelectbox"],
    div[data-testid="stDateInput"] {
      background: rgba(4, 12, 28, 0.8);
      border: 1px solid rgba(93, 126, 165, 0.34);
      border-radius: 14px;
      padding: 0.35rem 0.6rem;
    }

    label, .stMarkdown p, .stText {color: #ccddf8 !important;}

    button[kind="secondary"], button[kind="primary"] {
      border-radius: 12px !important;
      border: 1px solid rgba(0, 217, 232, 0.45) !important;
      background: rgba(2, 17, 34, 0.88) !important;
      color: var(--cyan) !important;
      font-family: "Orbitron", sans-serif !important;
      letter-spacing: 1.2px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
      gap: 0.45rem;
      border-bottom: 1px solid rgba(89, 120, 156, 0.24);
    }
    .stTabs [role="tab"] {
      border: 1px solid rgba(89, 120, 156, 0.26);
      border-radius: 12px 12px 0 0;
      background: rgba(2, 10, 25, 0.6);
      color: #a9bedf !important;
      font-family: "Orbitron", sans-serif;
      letter-spacing: 1px;
      padding: 0.5rem 0.8rem;
    }
    .stTabs [aria-selected="true"] {
      color: var(--cyan) !important;
      border-color: rgba(0, 217, 232, 0.6);
      box-shadow: inset 0 0 20px rgba(0, 217, 232, 0.12);
    }

    div[data-testid="stDataFrame"] {
      border: 1px solid rgba(89, 120, 156, 0.3);
      border-radius: 12px;
      background: rgba(2, 10, 25, 0.8);
    }

    @media (max-width: 900px) {
      .hero-title {font-size: 1.65rem;}
      .topbar {flex-direction: column; align-items: flex-start; gap: 0.6rem;}
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

    dealership_base = df["dealership"].astype(str).fillna("").str.split(",").str[0].str.strip()
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
            df.groupby("dealership", dropna=False).size().reset_index(name="sales_count").sort_values("sales_count", ascending=False)
        ),
        "sales_by_model": (
            df.groupby("model", dropna=False).size().reset_index(name="sales_count").sort_values("sales_count", ascending=False)
        ),
        "sales_by_branch": (
            df.groupby("branch_office", dropna=False).size().reset_index(name="sales_count").sort_values("sales_count", ascending=False)
        ),
        "dealership_model_branch": (
            df.groupby(["dealership", "model", "branch_office"], dropna=False).size().reset_index(name="sales_count").sort_values("sales_count", ascending=False)
        ),
    }

    dated_df = df[df["disbursed_date"].notna()].copy()
    if not dated_df.empty:
        dated_df["month"] = dated_df["disbursed_date"].dt.to_period("M").astype(str)
        dated_df["week_start"] = (
            dated_df["disbursed_date"] - pd.to_timedelta(dated_df["disbursed_date"].dt.dayofweek, unit="D")
        ).dt.date.astype(str)

        reports["sales_by_month"] = (
            dated_df.groupby("month", dropna=False).size().reset_index(name="sales_count").sort_values("month", ascending=False)
        )
        reports["sales_by_week"] = (
            dated_df.groupby("week_start", dropna=False).size().reset_index(name="sales_count").sort_values("week_start", ascending=False)
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

    return dated_only[(dated_only["disbursed_date"] >= start) & (dated_only["disbursed_date"] <= end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]


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
      <div class="status-wrap">
        <div class="pill">[ 0X-WATU-AUDIT ]</div>
        <div class="pill active">ACTIVE_NODE</div>
      </div>
    </div>

    <div class="hero">
      <div class="sync">SYNCHRONIZED_ACTIVE</div>
      <h2 class="hero-title">WATU DAILY <span>SALES</span></h2>
      <div class="cmd-row">
        <div class="cmd">ARCHIVES</div>
        <div class="cmd">UPLINK NEW DATA</div>
        <div class="cmd highlight">DISPATCH HUB</div>
        <div class="cmd">ARCHIVE PDF</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### <span class='section-title'>UPLINK DATA</span>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to activate the dashboard.")
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
        "Expected source headers: Client Mobile No, Branch Office, Dealership, disbursedon_date, registration_no, chasis_no, make, model, type"
    )
    st.stop()

prepared = prepare_data(df)
if prepared.empty:
    st.warning("No rows found where type is 'Mobile device'.")
    st.stop()

has_valid_dates = prepared["disbursed_date"].notna().any()

st.markdown("### <span class='section-title'>FILTER MATRIX</span>", unsafe_allow_html=True)
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

st.markdown("### <span class='section-title'>SYSTEM SNAPSHOT</span>", unsafe_allow_html=True)
mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.markdown("<div class='border-cyan'>", unsafe_allow_html=True)
    st.metric("TOTAL UNITS", int(len(filtered)))
    st.markdown("</div>", unsafe_allow_html=True)
with mc2:
    st.markdown("<div class='border-lime'>", unsafe_allow_html=True)
    top_dealer = reports["sales_by_dealership"].iloc[0]["dealership"] if not reports["sales_by_dealership"].empty else "N/A"
    st.metric("DOMINANT NODE", top_dealer)
    st.markdown("</div>", unsafe_allow_html=True)
with mc3:
    st.markdown("<div class='border-pink'>", unsafe_allow_html=True)
    st.metric("VALIDATION QUEUE", 0)
    st.markdown("</div>", unsafe_allow_html=True)
with mc4:
    st.markdown("<div class='border-amber'>", unsafe_allow_html=True)
    top_branch = reports["sales_by_branch"].iloc[0]["branch_office"] if not reports["sales_by_branch"].empty else "N/A"
    st.metric("STRATEGIC HUB", top_branch)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### <span class='section-title'>EXPORT MATRIX</span>", unsafe_allow_html=True)
col_export1, col_export2 = st.columns([1, 1])
col_export1.download_button(
    "EXPORT FULL REPORT (EXCEL)",
    data=to_excel_download(reports),
    file_name="management_sales_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
col_export2.download_button(
    "EXPORT FILTERED RAW DATA (CSV)",
    data=to_csv_download(filtered),
    file_name="filtered_mobile_device_data.csv",
    mime="text/csv",
)

st.markdown("### <span class='section-title'>MARKET CONCENTRATION</span>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["Dealership", "Model", "Branch", "Detailed"])
with tab1:
    show_report_section("Sales by Dealership", "sales_by_dealership", reports, "sales_by_dealership.csv")
with tab2:
    show_report_section("Sales by Model", "sales_by_model", reports, "sales_by_model.csv")
with tab3:
    show_report_section("Sales by Branch", "sales_by_branch", reports, "sales_by_branch.csv")
with tab4:
    show_report_section("Sales by Dealership + Model + Branch", "dealership_model_branch", reports, "sales_by_dealership_model_branch.csv")

if "sales_by_month" in reports or "sales_by_week" in reports:
    st.markdown("### <span class='section-title'>TIME TRENDS</span>", unsafe_allow_html=True)
    time_col1, time_col2 = st.columns(2)
    if "sales_by_month" in reports:
        with time_col1:
            show_report_section("Sales by Month", "sales_by_month", reports, "sales_by_month.csv")
    if "sales_by_week" in reports:
        with time_col2:
            show_report_section("Sales by Week", "sales_by_week", reports, "sales_by_week.csv")
