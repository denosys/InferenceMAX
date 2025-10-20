#!/usr/bin/env python3
# generate_html.py (sterilized)
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "docs"
DATA_DIR = OUT_DIR / "data"
OLDER_DIR = DATA_DIR / "older"
OUT_FILE = OUT_DIR / "index.html"
DIAG_FILE = OUT_DIR / "diagnostics.txt"
STATIC_DIR = OUT_DIR / "static"
TMP_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "generate_html_tmp"

# ensure output dirs exist (but do not create or modify DATA_DIR contents)
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

def list_data_files() -> List[Path]:
    """List files in docs/data excluding docs/data/older. Do not modify files."""
    if not DATA_DIR.exists():
        return []
    files = []
    for p in sorted(DATA_DIR.iterdir()):
        if p == OLDER_DIR:
            continue
        if p.is_file():
            files.append(p)
    return files

def load_json_safe(p: Path) -> Optional[Any]:
    try:
        return json.loads(p.read_bytes().decode("utf-8"))
    except Exception:
        try:
            return json.load(p.open("r", encoding="utf-8"))
        except Exception:
            return None

def normalize_records_from_json(j: Any) -> List[Dict[str, Any]]:
    if j is None:
        return []
    if isinstance(j, list):
        return [item for item in j if isinstance(item, dict)]
    if isinstance(j, dict):
        for key in ("results","data","records","items"):
            if key in j and isinstance(j[key], list):
                return [item for item in j[key] if isinstance(item, dict)]
        return [j]
    return []

def build_dataframe_from_files(files: List[Path]) -> pd.DataFrame:
    records = []
    for p in files:
        j = load_json_safe(p)
        if j is None:
            print(f"Warning: failed to parse {p.name}, skipping")
            continue
        recs = normalize_records_from_json(j)
        records.extend(recs)
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records)

def choose_metric_column(df: pd.DataFrame) -> Optional[str]:
    pref = ["value","metric","score"]
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
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines+markers",name=str(hw)))
    fig.update_layout(title=f"Sample: {metric}", margin=dict(t=60))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def build_plotly_html(df: pd.DataFrame) -> str:
    if df.empty:
        sample = [
            {"hardware":"cpu-a","value":12.3,"run_id":1},
            {"hardware":"cpu-a","value":13.1,"run_id":2},
            {"hardware":"gpu-x","value":9.8,"run_id":1},
            {"hardware":"gpu-x","value":10.2,"run_id":2},
        ]
        sample_df = pd.json_normalize(sample)
        html_sample = _build_sample_plot_html(sample_df, metric="value")
        info = (
            "<div style='font-family:system-ui,Arial,sans-serif;padding:24px;'>"
            "<h2>No data found in docs/data/</h2>"
            "<p>A sample chart is shown for layout/testing.</p>"
            "</div>"
        )
        return f"<!doctype html><html><head><meta charset='utf-8'><title>InferenceMAX — Pages</title></head><body>{info}{html_sample}</body></html>"

    if "tp" in df.columns:
        try:
            df = df[df["tp"] == 1]
        except Exception:
            pass

    if df.empty:
        return "<html><body><h3>No data after tp==1 filter</h3></body></html>"

    hw_candidates = ["hardware","device","platform","target"]
    hw_col = next((c for c in hw_candidates if c in df.columns), None)
    if hw_col is None:
        non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        hw_col = non_numeric[0] if non_numeric else None
    if hw_col is None:
        df["hardware"] = df.index.astype(str)
    else:
        df["hardware"] = df[hw_col].astype(str)

    metric = choose_metric_column(df)
    if metric is None:
        return "<html><body><h3>No numeric metric found in data</h3></body></html>"

    hardware_list = sorted(df["hardware"].dropna().unique())
    fig = go.Figure()
    x_col = next((c for c in ("timestamp","date","run_id","id","step") if c in df.columns), None)

    for hw in hardware_list:
        d = df[df["hardware"] == hw].copy()
        x = d[x_col].tolist() if x_col and x_col in d.columns else list(range(len(d)))
        y = d[metric].tolist()
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines+markers",name=str(hw)))

    buttons = []
    for i, hw in enumerate(hardware_list):
        vis = [False]*len(hardware_list)
        vis[i] = True
        buttons.append(dict(label=str(hw), method="update", args=[{"visible":vis},{"title":f"{metric} — {hw}"}]))
    buttons.insert(0, dict(label="All", method="update", args=[{"visible":[True]*len(hardware_list)}, {"title":f"{metric} — All hardware"}]))

    fig.update_layout(updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.12, xanchor="left", yanchor="top")],
                      title=f"{metric} — All hardware", margin=dict(t=80))
    return fig.to_html(full_html=True, include_plotlyjs="cdn")

def write_diagnostics(zip_files: List[str], files_in_data: List[str], records_count: int, sample_record: Optional[dict]):
    try:
        with DIAG_FILE.open("w", encoding="utf-8") as d:
            d.write(f"zip_files_present_in_repo_dir (not used): {zip_files}\n")
            d.write(f"files_in_docs_data (read-only): {files_in_data}\n")
            d.write(f"records_count: {records_count}\n")
            d.write("sample_record:\n")
            if sample_record:
                d.write(json.dumps(sample_record, indent=2))
            else:
                d.write("NONE\n")
        print(f"Wrote diagnostics: {DIAG_FILE}")
    except Exception as e:
        print(f"Warning: could not write diagnostics: {e}")

def main():
    # Note: Do NOT extract or modify data_zips/ or docs/data.
    zip_paths = sorted(p.name for p in (ROOT.glob("data_zips/*.zip")))
    data_files = list_data_files()
    data_file_names = [p.name for p in data_files]
    print(f"INFO: data files discovered (read-only): {data_file_names}")

    df = build_dataframe_from_files(data_files)
    records_count = len(df) if not df.empty else 0
    sample_record = None
    if records_count > 0:
        sample_record = df.iloc[0].to_dict()

    # write diagnostics
    write_diagnostics(zip_paths, data_file_names, records_count, sample_record)

    # write summary.json in docs/static (never touch docs/data)
    try:
        summary = {"files": data_file_names, "records_count": records_count}
        (STATIC_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote static summary: {STATIC_DIR/'summary.json'}")
    except Exception as e:
        print(f"Warning: could not write static summary: {e}")

    # build and write HTML
    html = build_plotly_html(df)
    try:
        OUT_FILE.write_text(html, encoding="utf-8")
        print(f"Wrote {OUT_FILE}")
    except Exception as e:
        print(f"Error: could not write {OUT_FILE}: {e}")

if __name__ == "__main__":
    main()
