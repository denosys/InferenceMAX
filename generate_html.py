#!/usr/bin/env python3
# generate_html.py
# English comments: this script extracts JSON files from zip artifacts in data_zips/,
# writes each JSON to InferenceMAX/docs/data/<zipname>__<filepath>.json,
# writes a data_index.json listing saved files, creates diagnostics.txt and docs/index.html.
# Requirements: pandas, plotly

import os
import json
import zipfile
import glob
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Configuration (change if needed)
INPUT_DIR = "data_zips"    # folder where the Action saves zip artifacts
OUT_DIR = "docs"           # folder served by GitHub Pages
DATA_DIR = os.path.join(OUT_DIR, "data")
OUT_FILE = os.path.join(OUT_DIR, "index.html")
DATA_INDEX = os.path.join(DATA_DIR, "data_index.json")

# Ensure directories exist
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def extract_jsons_from_zip(path: str) -> List[Dict[str, Any]]:
    """Extract all JSON files from a zip and return list of tuples (member_name, obj)."""
    results: List[Dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path, "r") as z:
            for name in z.namelist():
                if name.lower().endswith(".json"):
                    try:
                        raw = z.read(name)
                        j = json.loads(raw.decode("utf-8"))
                        results.append({"member": name, "obj": j})
                    except Exception as e:
                        print(f"Warning: could not read {name} in {path}: {e}")
    except zipfile.BadZipFile:
        print(f"Warning: bad zip file: {path}")
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
    return []


def safe_filename(s: str) -> str:
    """Make a filesystem-safe filename segment."""
    # replace path separators and spaces
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def save_extracted_jsons(zip_path: str) -> List[str]:
    """Extract JSONs from zip and save each as a file in DATA_DIR. Return list of saved filenames (relative to DATA_DIR)."""
    saved = []
    zname = Path(zip_path).stem
    extracted = extract_jsons_from_zip(zip_path)
    for item in extracted:
        member = item["member"]
        obj = item["obj"]
        # Build unique filename: <zipname>__<memberpath>.json
        member_safe = safe_filename(member)
        fname = f"{safe_filename(zname)}__{member_safe}"
        if not fname.lower().endswith(".json"):
            fname = fname + ".json"
        out_path = Path(DATA_DIR) / fname
        # write atomically
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            tmp.replace(out_path)
            saved.append(fname)
            print(f"Wrote extracted JSON: {out_path}")
        except Exception as e:
            print(f"Warning: cannot write {out_path}: {e}")
    return saved


def load_all_records_from_saved_jsons(saved_files: List[str]) -> List[Dict[str, Any]]:
    """Read saved JSON files and aggregate records for dataframe creation."""
    all_records: List[Dict[str, Any]] = []
    for fname in saved_files:
        path = Path(DATA_DIR) / fname
        try:
            with path.open("r", encoding="utf-8") as f:
                j = json.load(f)
            recs = normalize_records_from_json(j)
            all_records.extend(recs)
        except Exception as e:
            print(f"Warning: cannot read saved json {path}: {e}")
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


def _build_sample_plot_html(df: pd.DataFrame, metric: str) -> str:
    fig = go.Figure()
    hw_list = sorted(df["hardware"].unique())
    for hw in hw_list:
        d = df[df["hardware"] == hw]
        x = d["run_id"].tolist() if "run_id" in d.columns else list(range(len(d)))
        y = d[metric].tolist()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=str(hw)))
    fig.update_layout(title=f"Sample: {metric}", margin=dict(t=60))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_plotly_html(df: pd.DataFrame) -> str:
    """Build an interactive Plotly HTML page from the DataFrame."""
    if df.empty:
        sample = [
            {"hardware": "cpu-a", "value": 12.3, "run_id": 1},
            {"hardware": "cpu-a", "value": 13.1, "run_id": 2},
            {"hardware": "gpu-x", "value": 9.8, "run_id": 1},
            {"hardware": "gpu-x", "value": 10.2, "run_id": 2},
        ]
        sample_df = pd.json_normalize(sample)
        html_sample = _build_sample_plot_html(sample_df, metric="value")
        info = (
            "<div style='font-family:system-ui,Arial,sans-serif;padding:24px;'>"
            "<h2>No data found in data_zips/</h2>"
            "<p>The generator did not find JSON data inside the zip artifacts. Debug info is printed in the workflow logs.</p>"
            "<p>If you want to test the page layout, a sample chart is shown below.</p>"
            "</div>"
        )
        return f"<!doctype html><html><head><meta charset='utf-8'><title>InferenceMAX — Pages</title></head><body>{info}{html_sample}</body></html>"

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
        buttons.append(
            dict(
                label=str(hw),
                method="update",
                args=[{"visible": vis}, {"title": f"{metric} — {hw}"}],
            )
        )
    buttons.insert(
        0,
        dict(
            label="All",
            method="update",
            args=[{"visible": [True] * len(hardware_list)}, {"title": f"{metric} — All hardware"}],
        ),
    )

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.12, xanchor="left", yanchor="top")],
        title=f"{metric} — All hardware",
        margin=dict(t=80),
    )

    return fig.to_html(full_html=True, include_plotlyjs="cdn")


def main():
    # Find all zip files
    zip_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.zip")))
    print(f"DEBUG: zip files found in {INPUT_DIR}: {zip_paths}")

    # Extract and save JSONs from zips
    saved_files_all: List[str] = []
    for zpath in zip_paths:
        saved = save_extracted_jsons(zpath)
        saved_files_all.extend(saved)
    saved_files_all = sorted(set(saved_files_all))  # unique & sorted

    print(f"DEBUG: saved json files: {saved_files_all}")

    # If we saved anything, write data_index.json for client-side discovery
    try:
        with open(DATA_INDEX, "w", encoding="utf-8") as idxf:
            json.dump({"files": saved_files_all}, idxf, ensure_ascii=False, indent=2)
        print(f"Wrote data index: {DATA_INDEX}")
    except Exception as e:
        print(f"Warning: could not write data index: {e}")

    # Aggregate records from saved jsons (so the html generator uses the same data)
    records = load_all_records_from_saved_jsons(saved_files_all)
    df = to_dataframe(records)
    rows = len(df)
    print(f"DEBUG: Records loaded: {len(records)}. DataFrame rows: {rows}.")

    # Write diagnostics
    diag_path = os.path.join(OUT_DIR, "diagnostics.txt")
    try:
        with open(diag_path, "w", encoding="utf-8") as d:
            zips = sorted(zip_paths)
            d.write(f"zip_files_found: {zips}\n")
            d.write(f"saved_files: {saved_files_all}\n")
            d.write(f"records_count: {len(records)}\n")
            d.write("sample_record:\n")
            if records:
                d.write(json.dumps(records[0], indent=2))
            else:
                d.write("NONE\n")
        print(f"Wrote diagnostic file: {diag_path}")
    except Exception as e:
        print(f"Warning: could not write diagnostics: {e}")

    # Build and write HTML
    html = build_plotly_html(df)
    try:
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Wrote {OUT_FILE}.")
    except Exception as e:
        print(f"Error: could not write {OUT_FILE}: {e}")


if __name__ == "__main__":
    main()
