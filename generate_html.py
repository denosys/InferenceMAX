#!/usr/bin/env python3
# generate_html.py
# Requirements: pandas, plotly
# Usage: put .zip files into data_zips/, this script generates docs/index.html

import os
import io
import json
import zipfile
import glob
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go

# Configuration (change if needed)
INPUT_DIR = "data_zips"    # folder where the Action saves zip artifacts
OUT_DIR = "docs"           # folder served by GitHub Pages
OUT_FILE = os.path.join(OUT_DIR, "index.html")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

def extract_jsons_from_zip(path: str) -> List[Dict[str, Any]]:
    """Extract all JSON files from a zip and return them as Python objects."""
    results = []
    with zipfile.ZipFile(path, "r") as z:
        for name in z.namelist():
            if name.lower().endswith(".json"):
                try:
                    raw = z.read(name)
                    j = json.loads(raw.decode("utf-8"))
                    results.append(j)
                except Exception as e:
                    print(f"Warning: could not read {name} in {path}: {e}")
    return results

def normalize_records_from_json(j: Any) -> List[Dict[str, Any]]:
    """Normalize various JSON shapes into a list of record dicts."""
    if j is None:
        return []
    if isinstance(j, list):
        return [item for item in j if isinstance(item, dict)]
    if isinstance(j, dict):
        # common cases: {'results': [...]}, {'data': [...]}
        for key in ("results", "data", "records", "items"):
            if key in j and isinstance(j[key], list):
                return [item for item in j[key] if isinstance(item, dict)]
        # otherwise return the dict as a single record
        return [j]
    # other types (e.g. string) -> no records
    return []

def load_all_records(input_dir: str) -> List[Dict[str, Any]]:
    """Load and aggregate records from all JSONs contained in zip files."""
    all_records: List[Dict[str, Any]] = []
    zip_paths = sorted(glob.glob(os.path.join(input_dir, "*.zip")))
    for zpath in zip_paths:
        json_objs = extract_jsons_from_zip(zpath)
        for j in json_objs:
            recs = normalize_records_from_json(j)
            all_records.extend(recs)
    return all_records

def to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of dicts to a pandas DataFrame using json_normalize."""
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records)
    return df

def choose_metric_column(df: pd.DataFrame) -> Optional[str]:
    """Choose a numeric metric column: prefer 'value', 'metric', 'score', else first numeric."""
    pref = ["value", "metric", "score"]
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    for p in pref:
        if p in numeric:
            return p
        if p in df.columns and pd.api.types.is_numeric_dtype(df[p]):
            return p
    return numeric[0] if numeric else None

def build_plotly_html(df: pd.DataFrame) -> str:
    """Build an interactive Plotly HTML page from the DataFrame."""
    if df.empty:
        return "<html><body><h3>No data found</h3></body></html>"

    # Filter tp==1 when present
    if "tp" in df.columns:
        try:
            df = df[df["tp"] == 1]
        except Exception:
            pass

    if df.empty:
        return "<html><body><h3>No data after tp==1 filter</h3></body></html>"

    # Ensure we have a hardware column: try common names, fallback to first non-numeric or index
    hw_candidates = ["hardware", "device", "platform", "target"]
    hw_col = next((c for c in hw_candidates if c in df.columns), None)
    if hw_col is None:
        non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        hw_col = non_numeric[0] if non_numeric else None
    if hw_col is None:
        df["hardware"] = df.index.astype(str)
        hw_col = "hardware"
    else:
        df["hardware"] = df[hw_col].astype(str)

    metric = choose_metric_column(df)
    if metric is None:
        return "<html><body><h3>No numeric metric found in data</h3></body></html>"

    hardware_list = sorted(df["hardware"].dropna().unique())

    fig = go.Figure()
    # Prefer a time/run column for x-axis if available
    x_col = None
    for c in ("timestamp", "date", "run_id", "id", "step"):
        if c in df.columns:
            x_col = c
            break

    for hw in hardware_list:
        d = df[df["hardware"] == hw].copy()
        if x_col and x_col in d.columns:
            x = d[x_col].tolist()
        else:
            x = list(range(len(d)))
        y = d[metric].tolist()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=str(hw)))

    # Dropdown to select single hardware or All
    buttons = []
    for i, hw in enumerate(hardware_list):
        vis = [False] * len(hardware_list)
        vis[i] = True
        buttons.append(dict(label=str(hw),
                            method="update",
                            args=[{"visible": vis},
                                  {"title": f"{metric} — {hw}"}]))
    buttons.insert(0, dict(label="All",
                           method="update",
                           args=[{"visible": [True]*len(hardware_list)},
                                 {"title": f"{metric} — All hardware"}]))

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.12, xanchor="left", yanchor="top")],
        title=f"{metric} — All hardware",
        margin=dict(t=80)
    )

    # Use CDN for plotly.js to keep page light
    return fig.to_html(full_html=True, include_plotlyjs="cdn")

def main():
    records = load_all_records(INPUT_DIR)
    df = to_dataframe(records)
    html = build_plotly_html(df)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    rows = len(df)
    print(f"Wrote {OUT_FILE}. Records loaded: {len(records)}. DataFrame rows: {rows}.")

if __name__ == "__main__":
    main()
