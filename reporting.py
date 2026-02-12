import hashlib
import json
import os
import re
import smtplib
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO, StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
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
PRESETS_FILE = os.path.join(APPDATA_DIR, "filter_presets.json")
AUDIT_FILE = os.path.join(APPDATA_DIR, "audit_log.json")


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
    if not os.path.exists(PRESETS_FILE):
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    if not os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, "w", encoding="utf-8") as f:
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


def load_filter_presets() -> List[dict]:
    ensure_storage()
    with open(PRESETS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_filter_preset(name: str, user: str, payload: dict) -> None:
    ensure_storage()
    presets = load_filter_presets()
    preset = {
        "id": hashlib.sha1(f"{name}_{user}_{datetime.now().isoformat()}".encode("utf-8")).hexdigest()[:10],
        "name": name,
        "user": user,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "payload": payload,
    }
    presets.insert(0, preset)
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets[:100], f, indent=2)


def delete_filter_preset(preset_id: str) -> None:
    presets = [p for p in load_filter_presets() if p.get("id") != preset_id]
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2)


def log_audit_event(user: str, action: str, details: dict) -> None:
    ensure_storage()
    with open(AUDIT_FILE, "r", encoding="utf-8") as f:
        items = json.load(f)
    items.insert(
        0,
        {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user,
            "action": action,
            "details": details,
        },
    )
    with open(AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump(items[:300], f, indent=2)


def load_audit_log() -> List[dict]:
    ensure_storage()
    with open(AUDIT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


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


def data_dictionary() -> pd.DataFrame:
    rows = [
        ("Client Mobile No", "Text", "Customer contact number", "optional"),
        ("Branch Office", "Text", "Branch/office name", "required"),
        ("Dealership", "Text", "Dealer name (text before comma is grouped)", "required"),
        ("disbursedon_date", "Date", "Disbursal date", "recommended"),
        ("registration_no", "Text", "Registration number", "optional"),
        ("chasis_no", "Text", "Chassis number", "optional"),
        ("make", "Text", "Manufacturer", "optional"),
        ("model", "Text", "Product model", "required"),
        ("type", "Text", "Only 'Mobile device' rows are counted", "required"),
    ]
    return pd.DataFrame(rows, columns=["column", "type", "description", "rule"])


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    dated = df[df["disbursed_date"].notna()].copy()
    if dated.empty:
        return pd.DataFrame(columns=["period", "sales_count", "zscore", "severity"])

    weekly = (
        dated.assign(period=dated["disbursed_date"].dt.to_period("W").astype(str))
        .groupby("period")
        .size()
        .reset_index(name="sales_count")
    )
    if weekly.shape[0] < 3:
        return pd.DataFrame(columns=["period", "sales_count", "zscore", "severity"])

    mean = weekly["sales_count"].mean()
    std = weekly["sales_count"].std(ddof=0)
    if std == 0:
        return pd.DataFrame(columns=["period", "sales_count", "zscore", "severity"])

    weekly["zscore"] = (weekly["sales_count"] - mean) / std
    out = weekly[weekly["zscore"].abs() >= 1.8].copy()
    out["severity"] = np.where(out["zscore"].abs() >= 2.5, "high", "medium")
    return out.sort_values("zscore", ascending=False)


def forecast_sales(df: pd.DataFrame, by: str = "dealership", periods: int = 3) -> pd.DataFrame:
    if by not in {"dealership", "branch_office", "model"}:
        by = "dealership"

    dated = df[df["disbursed_date"].notna()].copy()
    if dated.empty:
        return pd.DataFrame(columns=[by, "forecast_next_period"])

    dated["period"] = dated["disbursed_date"].dt.to_period("M")
    grouped = dated.groupby([by, "period"]).size().reset_index(name="sales_count")
    results = []

    for key, g in grouped.groupby(by):
        g = g.sort_values("period")
        y = g["sales_count"].to_numpy(dtype=float)
        x = np.arange(len(y), dtype=float)
        if len(y) == 1:
            pred = y[-1]
        else:
            slope, intercept = np.polyfit(x, y, 1)
            pred = intercept + slope * (len(y) + periods - 1)
        results.append({by: key, "forecast_next_period": max(round(float(pred)), 0)})

    return pd.DataFrame(results).sort_values("forecast_next_period", ascending=False)


def auto_insights(reports: Dict[str, pd.DataFrame], comparison: Dict[str, float], quality: DataQuality) -> List[str]:
    insights: List[str] = []
    if "sales_by_dealership" in reports and not reports["sales_by_dealership"].empty:
        top = reports["sales_by_dealership"].iloc[0]
        insights.append(f"Top dealership is {top['dealership']} with {int(top['sales_count'])} sales.")
    if "sales_by_branch" in reports and not reports["sales_by_branch"].empty:
        top = reports["sales_by_branch"].iloc[0]
        insights.append(f"Highest volume branch is {top['branch_office']} ({int(top['sales_count'])} sales).")
    if "sales_by_model" in reports and not reports["sales_by_model"].empty:
        top = reports["sales_by_model"].iloc[0]
        insights.append(f"Best performing model is {top['model']} with {int(top['sales_count'])} units.")
    insights.append(f"Period-over-period movement is {comparison['pct_change']:.1f}%.")
    if quality.duplicate_rows > 0 or quality.invalid_dates > 0:
        insights.append(
            f"Data quality risk: {quality.duplicate_rows} duplicate rows and {quality.invalid_dates} invalid dates."
        )
    return insights


def root_cause_suggestions(df: pd.DataFrame) -> List[str]:
    suggestions: List[str] = []
    comp = compare_periods(df)
    if comp["pct_change"] < 0:
        suggestions.append("Decline detected: review branch-wise contribution for underperforming locations.")
        suggestions.append("Check model mix shift: lower share of high-volume models can reduce totals.")
        suggestions.append("Inspect date coverage and delayed disbursals in current period.")
    else:
        suggestions.append("Growth is positive: replicate top-branch playbook across low-performing branches.")
        suggestions.append("Prioritize inventory for high-conversion models to sustain trend.")
    return suggestions


def smart_alerts(
    reports: Dict[str, pd.DataFrame], quality: DataQuality, comparison: Dict[str, float], drop_threshold_pct: float = -10.0
) -> List[dict]:
    alerts: List[dict] = []
    if comparison["pct_change"] <= drop_threshold_pct:
        alerts.append({"level": "high", "message": f"Sales dropped {comparison['pct_change']:.1f}% vs previous period."})
    if quality.invalid_dates > 0:
        alerts.append({"level": "medium", "message": f"{quality.invalid_dates} rows have invalid disbursed dates."})
    if quality.duplicate_rows > 0:
        alerts.append({"level": "medium", "message": f"{quality.duplicate_rows} duplicate records detected."})
    if "sales_by_branch" in reports and not reports["sales_by_branch"].empty:
        total = reports["sales_by_branch"]["sales_count"].sum()
        top = reports["sales_by_branch"].iloc[0]
        if total > 0 and (top["sales_count"] / total) > 0.5:
            alerts.append(
                {
                    "level": "low",
                    "message": f"Concentration risk: {top['branch_office']} contributes over 50% of sales.",
                }
            )
    return alerts


def cleaning_assistant(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    issues = []
    if "dealership" in df.columns:
        raw = df["dealership"].astype(str)
        normalized = raw.str.split(",").str[0].str.strip().str.lower()
        mapping = pd.DataFrame({"raw": raw, "normalized": normalized}).drop_duplicates()
        suggestions = mapping[mapping["raw"].str.lower() != mapping["normalized"]]
    else:
        suggestions = pd.DataFrame(columns=["raw", "normalized"])

    if "disbursedon_date" in df.columns:
        parsed = pd.to_datetime(df["disbursedon_date"], errors="coerce")
        bad_dates = df[parsed.isna()].head(50)
        if not bad_dates.empty:
            issues.append(("invalid_date", bad_dates))

    dupes = df[df.duplicated()].head(50)
    if not dupes.empty:
        issues.append(("duplicates", dupes))

    issue_rows = []
    for issue_type, issue_df in issues:
        issue_rows.append({"issue": issue_type, "count": int(issue_df.shape[0])})
    summary = pd.DataFrame(issue_rows, columns=["issue", "count"])

    return {"summary": summary, "dealership_suggestions": suggestions.head(200)}


def target_vs_actual(df: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
    required = {"dealership", "branch_office", "target_sales"}
    if not required.issubset(set(targets_df.columns)):
        return pd.DataFrame(columns=["dealership", "branch_office", "target_sales", "actual_sales", "achievement_pct"])

    actual = (
        df.groupby(["dealership", "branch_office"])
        .size()
        .reset_index(name="actual_sales")
    )
    merged = targets_df.copy()
    merged["target_sales"] = pd.to_numeric(merged["target_sales"], errors="coerce").fillna(0)
    merged = merged.merge(actual, on=["dealership", "branch_office"], how="left")
    merged["actual_sales"] = merged["actual_sales"].fillna(0)
    merged["achievement_pct"] = np.where(
        merged["target_sales"] > 0,
        (merged["actual_sales"] / merged["target_sales"]) * 100,
        0,
    )
    return merged.sort_values("achievement_pct", ascending=False)


def recommend_targets(df: pd.DataFrame) -> pd.DataFrame:
    fc = forecast_sales(df, by="dealership", periods=1)
    if fc.empty:
        return pd.DataFrame(columns=["dealership", "recommended_target"])
    fc["recommended_target"] = (fc["forecast_next_period"] * 1.08).round().astype(int)
    return fc[["dealership", "recommended_target"]].sort_values("recommended_target", ascending=False)


def nl_qa(question: str, df: pd.DataFrame, reports: Dict[str, pd.DataFrame]) -> str:
    q = question.strip().lower()
    if not q:
        return "Ask a question like: Which branch has highest sales this month?"

    if "top dealership" in q or "highest dealership" in q:
        top = reports["sales_by_dealership"].iloc[0]
        return f"Top dealership is {top['dealership']} with {int(top['sales_count'])} sales."

    if "top branch" in q or "highest branch" in q:
        top = reports["sales_by_branch"].iloc[0]
        return f"Top branch is {top['branch_office']} with {int(top['sales_count'])} sales."

    if "top model" in q or "highest model" in q:
        top = reports["sales_by_model"].iloc[0]
        return f"Top model is {top['model']} with {int(top['sales_count'])} sales."

    if "total sales" in q or "how many sales" in q:
        return f"Total sales in current view: {int(len(df))}."

    if "drop" in q or "decline" in q:
        comp = compare_periods(df)
        return f"Period change is {comp['pct_change']:.1f}% ({int(comp['current'])} vs {int(comp['previous'])})."

    return (
        "I can answer: top dealership, top branch, top model, total sales, or period decline. "
        "Try: 'Which is the top branch?'"
    )


def generate_narrative(
    reports: Dict[str, pd.DataFrame], comparison: Dict[str, float], alerts: List[dict], insights: List[str]
) -> str:
    lines = ["Executive Summary", ""]
    lines.append(f"- Period-over-period performance: {comparison['pct_change']:.1f}%.")
    lines.extend([f"- {ins}" for ins in insights[:4]])
    if alerts:
        lines.append("")
        lines.append("Risk Alerts")
        lines.extend([f"- [{a['level'].upper()}] {a['message']}" for a in alerts[:5]])
    return "\n".join(lines)


def what_if_simulation(base_total: int, dealership_uplift_pct: float, branch_uplift_pct: float) -> Dict[str, float]:
    effect = 1 + (dealership_uplift_pct / 100.0) + (branch_uplift_pct / 100.0)
    projected = max(int(round(base_total * effect)), 0)
    return {
        "base_total": float(base_total),
        "projected_total": float(projected),
        "delta": float(projected - base_total),
    }


def schedule_report_helper(frequency: str, day: str, time_24h: str, recipients: List[str]) -> str:
    recipient_str = ", ".join(recipients) if recipients else "no recipients provided"
    return (
        f"Suggested schedule: {frequency} on {day} at {time_24h}. "
        f"Send to: {recipient_str}. Configure with your scheduler/automation runtime."
    )
