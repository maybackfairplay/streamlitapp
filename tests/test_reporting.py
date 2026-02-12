import pandas as pd

from reporting import build_reports, map_columns, prepare_data


def test_prepare_filters_mobile_device_only():
    df = pd.DataFrame(
        {
            "type": ["Mobile device", "Accessory"],
            "dealership": ["ABC, Z1", "ABC, Z2"],
            "branch_office": ["N", "N"],
            "model": ["A", "A"],
            "make": ["M", "M"],
            "disbursedon_date": ["2026-01-01", "2026-01-01"],
        }
    )
    out = prepare_data(df)
    assert len(out) == 1
    assert out.iloc[0]["dealership"] == "ABC"


def test_map_columns_normalizes_headers():
    df = pd.DataFrame(columns=["Branch Office", "Dealership", "Model", "Type"])
    mapped = map_columns(df)
    assert "branch_office" in mapped.columns
    assert "dealership" in mapped.columns
    assert "model" in mapped.columns
    assert "type" in mapped.columns


def test_build_reports_counts_rows():
    df = pd.DataFrame(
        {
            "dealership": ["A", "A", "B"],
            "branch_office": ["N", "N", "S"],
            "model": ["X", "Y", "X"],
            "disbursed_date": pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-06"]),
        }
    )
    reports = build_reports(df)
    assert int(reports["sales_by_dealership"].loc[reports["sales_by_dealership"]["dealership"] == "A", "sales_count"].iloc[0]) == 2
