import hashlib
import json
import os
import re
import smtplib
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO, StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd

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

APPDATA_DIR = ".appdata"
UPLOADS_DIR = os.path.join(APPDATA_DIR, "uploads")
HISTORY_FILE = os.path.join(APPDATA_DIR, "upload_history.json")


@dataclass
class DataQuality:
    total_rows: int
    mobile_rows: int
    excluded_non_mobile: int
    invalid_dates: int
    missing_dealership: int
    missing_branch: int
    missing_model: int
    duplicate_rows: int


def normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_source = {normalize_col_name(c): c for c in df.columns}
    rename_map: Dict[str, str] = {}

    for canonical, normalized in CANONICAL_COLUMNS.items():
        if normalized in normalized_source:
            rename_map[normalized_source[normalized]] = canonical

    return df.rename(columns=rename_map)


def validate_required_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in REQUIRED_FOR_REPORT if c not in df.columns]


def data_quality_summary(df: pd.DataFrame) -> DataQuality:
    total_rows = len(df)
    type_col = df["type"].astype(str).str.strip().str.lower() if "type" in df.columns else pd.Series([], dtype=str)
    mobile_rows = int((type_col == "mobile device").sum()) if len(type_col) else 0

    parsed_dates = pd.to_datetime(df.get("disbursedon_date"), errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum()) if "disbursedon_date" in df.columns else total_rows

    missing_dealership = int(df.get("dealership", pd.Series([], dtype=str)).astype(str).str.strip().eq("").sum())
    missing_branch = int(df.get("branch_office", pd.Series([], dtype=str)).astype(str).str.strip().eq("").sum())
    missing_model = int(df.get("model", pd.Series([], dtype=str)).astype(str).str.strip().eq("").sum())
    duplicate_rows = int(df.duplicated().sum())

    return DataQuality(
        total_rows=total_rows,
        mobile_rows=mobile_rows,
        excluded_non_mobile=max(total_rows - mobile_rows, 0),
        invalid_dates=invalid_dates,
        missing_dealership=missing_dealership,
        missing_branch=missing_branch,
        missing_model=missing_model,
        duplicate_rows=duplicate_rows,
    )


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
    df["make"] = df.get("make", "").astype(str).str.strip()
    df["disbursed_date"] = pd.to_datetime(df.get("disbursedon_date"), errors="coerce")
    return df


def filter_data(
    df: pd.DataFrame,
    dealerships: List[str],
    branches: List[str],
    models: List[str],
    makes: List[str],
) -> pd.DataFrame:
    out = df.copy()
    if dealerships:
        out = out[out["dealership"].isin(dealerships)]
    if branches:
        out = out[out["branch_office"].isin(branches)]
    if models:
        out = out[out["model"].isin(models)]
    if makes:
        out = out[out["make"].isin(makes)]
    return out


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
            dated_df["disbursed_date"] - pd.to_timedelta(dated_df["disbursed_date"].dt.dayofweek, unit="D")
        ).dt.date.astype(str)

        reports["sales_by_month"] = (
            dated_df.groupby("month", dropna=False).size().reset_index(name="sales_count").sort_values("month", ascending=True)
        )
        reports["sales_by_week"] = (
            dated_df.groupby("week_start", dropna=False).size().reset_index(name="sales_count").sort_values("week_start", ascending=True)
        )

    return reports


def compare_periods(df: pd.DataFrame) -> Dict[str, float]:
    dated = df[df["disbursed_date"].notna()].copy()
    if dated.empty:
        return {"current": float(len(df)), "previous": 0.0, "pct_change": 0.0}

    max_date = dated["disbursed_date"].max().normalize()
    min_date = dated["disbursed_date"].min().normalize()
    span_days = max((max_date - min_date).days + 1, 1)
    period = max(span_days // 2, 1)

    current_start = max_date - pd.Timedelta(days=period - 1)
    prev_end = current_start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=period - 1)

    current = float(dated[(dated["disbursed_date"] >= current_start) & (dated["disbursed_date"] <= max_date)].shape[0])
    previous = float(dated[(dated["disbursed_date"] >= prev_start) & (dated["disbursed_date"] <= prev_end)].shape[0])

    if previous == 0:
        pct = 100.0 if current > 0 else 0.0
    else:
        pct = ((current - previous) / previous) * 100.0

    return {"current": current, "previous": previous, "pct_change": pct}


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


def to_pdf_download(reports: Dict[str, pd.DataFrame], comparison: Dict[str, float], total_sales: int) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        # If reportlab is unavailable in runtime, return a plain-text fallback payload.
        text = (
            "Management Sales Briefing\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Total Mobile Device Sales: {total_sales}\n"
            f"Comparison: {int(comparison['current'])} vs {int(comparison['previous'])} ({comparison['pct_change']:.1f}%)\n"
        )
        return text.encode("utf-8")

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Management Sales Briefing")

    y -= 28
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    y -= 28
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"Total Mobile Device Sales: {total_sales}")

    y -= 20
    c.setFont("Helvetica", 11)
    c.drawString(
        40,
        y,
        f"Comparison (Current vs Previous): {int(comparison['current'])} vs {int(comparison['previous'])} ({comparison['pct_change']:.1f}%)",
    )

    y -= 28
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Top Insights")

    insights: List[Tuple[str, str]] = []
    if "sales_by_dealership" in reports and not reports["sales_by_dealership"].empty:
        row = reports["sales_by_dealership"].iloc[0]
        insights.append(("Top Dealership", f"{row['dealership']} ({int(row['sales_count'])})"))
    if "sales_by_branch" in reports and not reports["sales_by_branch"].empty:
        row = reports["sales_by_branch"].iloc[0]
        insights.append(("Top Branch", f"{row['branch_office']} ({int(row['sales_count'])})"))
    if "sales_by_model" in reports and not reports["sales_by_model"].empty:
        row = reports["sales_by_model"].iloc[0]
        insights.append(("Top Model", f"{row['model']} ({int(row['sales_count'])})"))

    c.setFont("Helvetica", 11)
    for label, value in insights:
        y -= 20
        c.drawString(50, y, f"- {label}: {value}")

    c.showPage()
    c.save()
    return buffer.getvalue()


def ensure_storage() -> None:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)


def _read_history() -> List[dict]:
    ensure_storage()
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_history(items: List[dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(items[:50], f, indent=2)


def save_upload_snapshot(file_name: str, file_bytes: bytes) -> None:
    ensure_storage()
    digest = hashlib.sha1(file_bytes).hexdigest()[:12]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{ts}_{digest}_{os.path.basename(file_name)}"
    saved_path = os.path.join(UPLOADS_DIR, saved_name)

    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    rows = pd.read_csv(BytesIO(file_bytes), dtype=str).shape[0]
    history = _read_history()
    history.insert(
        0,
        {
            "id": f"{ts}_{digest}",
            "saved_name": saved_name,
            "original_name": file_name,
            "rows": int(rows),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    unique = []
    seen = set()
    for h in history:
        if h["id"] not in seen:
            seen.add(h["id"])
            unique.append(h)
    _write_history(unique)


def load_upload_history() -> List[dict]:
    return _read_history()


def load_snapshot_df(saved_name: str) -> pd.DataFrame:
    path = os.path.join(UPLOADS_DIR, saved_name)
    return pd.read_csv(path, dtype=str)


def send_email_report(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    sender: str,
    recipients: List[str],
    subject: str,
    body: str,
    attachment_name: str,
    attachment_bytes: bytes,
) -> Tuple[bool, str]:
    from email.message import EmailMessage

    if not recipients:
        return False, "No recipients provided."

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)
    msg.add_attachment(
        attachment_bytes,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=attachment_name,
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            server.starttls()
            if smtp_user:
                server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return True, "Email sent successfully."
    except Exception as exc:
        return False, f"Email failed: {exc}"
