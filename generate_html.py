#!/usr/bin/env python3
# generate_html.py (updated)
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

OUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# mapping filenames -> (model, isl, osl)
FILE_MAP = {
    "results_70b_1k1k": ("Llama-3.3-70B-Instruct", "1k", "1k"),
    "results_70b_8k1k": ("Llama-3.3-70B-Instruct", "8k", "1k"),
    "results_70b_1k8k": ("Llama-3.3-70B-Instruct", "1k", "8k"),
    "results_dsr1_1k1k": ("DeepSeek-R1-0528", "1k", "1k"),
    "results_dsr1_8k1k": ("DeepSeek-R1-0528", "8k", "1k"),
    "results_dsr1_1k8k": ("DeepSeek-R1-0528", "1k", "8k"),
    "results_gptoss_1k1k": ("gpt-oss-120b", "1k", "1k"),
    "results_gptoss_8k1k": ("gpt-oss-120b", "8k", "1k"),
    "results_gptoss_1k8k": ("gpt-oss-120b", "1k", "8k"),
}

def list_data_files() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    files = []
    for p in sorted(DATA_DIR.iterdir()):
        if p == OLDER_DIR:
            continue
        if p.is_file() and p.suffix.lower() == ".json":
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
        # attach inferred file key metadata if filename matches FILE_MAP
        key = next((k for k in FILE_MAP.keys() if k in p.name), None)
        for r in recs:
            if key:
                m, isl, osl = FILE_MAP[key]
                r.setdefault("model", m)
                r.setdefault("isl", isl)
                r.setdefault("osl", osl)
                r.setdefault("file_key", key)
        records.extend(recs)
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records)
    # normalize hw name
    if "hw" in df.columns:
        df["hw"] = df["hw"].astype(str).str.lower()
    elif "hardware" in df.columns:
        df["hw"] = df["hardware"].astype(str).str.lower()
    # ensure precision column
    if "precision" not in df.columns:
        df["precision"] = "fp8"
    # coerce numeric-like strings
    for c in df.columns:
        if df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().any():
                df[c] = coerced
    return df

def choose_metric_column(df: pd.DataFrame) -> Optional[str]:
    pref = ["tput_per_gpu","value","metric","score"]
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    for p in pref:
        if p in numeric:
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

def build_plotly_html(df: pd.DataFrame, data_file_names: List[str]) -> str:
    # construct client-side datasets mapping by file_key (only files matched in FILE_MAP)
    client_map = {}
    for key in FILE_MAP.keys():
        # find file present
        match = next((n for n in data_file_names if key in n), None)
        if match:
            p = DATA_DIR / match
            j = load_json_safe(p)
            recs = normalize_records_from_json(j)
            # normalize records same as server
            for r in recs:
                r.setdefault("model", FILE_MAP[key][0])
                r.setdefault("isl", FILE_MAP[key][1])
                r.setdefault("osl", FILE_MAP[key][2])
            # coerce numeric fields to strings where necessary for JSON
            client_map[key] = {
                "columns": list(pd.json_normalize(recs).columns),
                "records": recs
            }
    # default metric choices
    default_x = "median_e2el"
    # start building HTML + controls
    header = "<!doctype html><html><head><meta charset='utf-8'><title>InferenceMAX — Interactive</title>"
    header += "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"
    header += "<style>body{font-family:system-ui,Arial,sans-serif;margin:12px;} select{margin-right:8px;} .controls{margin-bottom:8px;}</style>"
    header += "</head><body>"
    controls = (
        "<div class='controls'>"
        "<label>Modello:</label><select id='model_sel'></select>"
        "<label>ISL/OSL:</label><select id='ctx_sel'></select>"
        "<label>Precisione:</label><select id='prec_sel'><option value='all'>Tutte</option><option value='fp8'>FP8</option><option value='fp4'>FP4</option></select>"
        "<label>TP (cards):</label><select id='tp_sel'><option value='all'>All</option></select>"
        "<label>Connetti TP:</label><select id='tp_line'><option value='yes'>Sì</option><option value='no' selected>No</option></select>"
        "<label>Asse Y:</label><select id='y_sel'></select>"
        "<label>Asse X:</label><select id='x_sel'><option value='median_e2el'>E2E latency</option><option value='median_intvty'>Interactivity</option></select>"
        "<label>Run/Artifact:</label><strong id='run_info'></strong>"
        "</div>"
    )
    plot_div = "<div id='plot_div' style='width:100%;height:700px;'></div>"
    # embed client data
    body_js = f"<script>const CLIENT_MAP = {json.dumps(client_map)}; const FILE_MAP = {json.dumps(FILE_MAP)};</script>"
    # main JS to populate controls and render plot (kept concise)
    main_js = """
<script>
const modelSel = document.getElementById('model_sel');
const ctxSel = document.getElementById('ctx_sel');
const precSel = document.getElementById('prec_sel');
const tpSel = document.getElementById('tp_sel');
const tpLine = document.getElementById('tp_line');
const ySel = document.getElementById('y_sel');
const xSel = document.getElementById('x_sel');
const runInfo = document.getElementById('run_info');

function buildModelsFromCLIENT(){
  const models = {};
  for (const k of Object.keys(CLIENT_MAP)){
    const recs = CLIENT_MAP[k].records || [];
    if (!recs.length) continue;
    const m = recs[0].model || 'unknown';
    const isl = recs[0].isl || '';
    const osl = recs[0].osl || '';
    models[m] = models[m] || [];
    models[m].push([isl, osl, k]);
  }
  return models;
}

const MODELS = buildModelsFromCLIENT();

function populateModels(){
  modelSel.innerHTML = '';
  for (const m of Object.keys(MODELS)){
    const opt = document.createElement('option'); opt.value = m; opt.text = m; modelSel.appendChild(opt);
  }
}
function populateContextsForModel(model){
  ctxSel.innerHTML = '';
  const entries = MODELS[model] || [];
  for (const e of entries){
    const opt = document.createElement('option'); opt.value = e[2]; opt.text = e[0] + '/' + e[1]; ctxSel.appendChild(opt);
  }
}
function updateRunInfo(key){
  runInfo.textContent = key || '';
}
function populateYOptionsForKey(key){
  ySel.innerHTML = '';
  const payload = CLIENT_MAP[key];
  if (!payload){ ySel.appendChild(new Option('(no data)','')); return; }
  const cols = payload.columns || [];
  // detect numeric by scanning first record
  const rec = (payload.records && payload.records.length) ? payload.records[0] : {};
  const numeric = [];
  for (const c of cols){
    const v = rec[c];
    if (v === undefined || v === null || v === '') continue;
    if (!isNaN(Number(v))) numeric.push(c);
  }
  if (numeric.includes('tput_per_gpu')) ySel.appendChild(new Option('tput_per_gpu','tput_per_gpu'));
  for (const c of numeric){
    if (c==='tput_per_gpu') continue;
    ySel.appendChild(new Option(c,c));
  }
  if (!ySel.options.length) ySel.appendChild(new Option('(no numeric)',''));
}
function populateTpOptionsForKey(key){
  tpSel.innerHTML = '';
  const payload = CLIENT_MAP[key];
  const seen = new Set();
  if (!payload || !payload.records){ tpSel.appendChild(new Option('All','all')); return; }
  for (const r of payload.records){
    const v = r['tp'];
    if (v===undefined || v===null || v==='') continue;
    seen.add(String(v));
  }
  tpSel.appendChild(new Option('All','all'));
  Array.from(seen).sort((a,b)=>Number(a)-Number(b)).forEach(v=>tpSel.appendChild(new Option(v,v)));
}

function buildTraces(records, xcol, ycol, connectTp, tpFilter, precFilter){
  const traces = [];
  if (!records || !records.length) return traces;
  let recs = records.slice();
  if (precFilter && precFilter!=='all') recs = recs.filter(r=> (r.precision||'').toString().toLowerCase()===precFilter);
  if (tpFilter && tpFilter!=='all') recs = recs.filter(r=> String(r.tp)===String(tpFilter));
  const groups = {};
  for (const r of recs){
    const hw = (r.hw||r.hardware||'unknown').toString().toLowerCase();
    groups[hw] = groups[hw]||[];
    groups[hw].push(r);
  }
  const hwKeys = Object.keys(groups).sort();
  for (const hw of hwKeys){
    const grp = groups[hw];
    const hasTp = grp.some(r=> r.hasOwnProperty('tp') && r.tp!=='');
    if (hasTp && connectTp){
      const tmap = {};
      for (const r of grp){
        const tp = (r.tp!==undefined && r.tp!=='')? String(r.tp): 'none';
        tmap[tp]=tmap[tp]||[];
        tmap[tp].push(r);
      }
      const tpKeys = Object.keys(tmap).sort((a,b)=>{ const na=Number(a), nb=Number(b); if(!isNaN(na)&&!isNaN(nb)) return na-nb; return a.localeCompare(b);});
      for (const tp of tpKeys){
        const rows = tmap[tp];
        const xs = rows.map(r=> Number(r[xcol])); const ys = rows.map(r=> Number(r[ycol]));
        traces.push({x:xs,y:ys,mode:'lines+markers',name: hw + ' tp=' + tp, legendgroup: hw,
                     text: rows.map(r=> 'tp=' + (r.tp||'') + ' model=' + (r.model||'')), hoverinfo:'text+x+y'});
      }
    } else {
      const xs = grp.map(r=> Number(r[xcol])); const ys = grp.map(r=> Number(r[ycol]));
      traces.push({x:xs,y:ys,mode:'markers',name: hw, legendgroup: hw,
                   text: grp.map(r=> 'tp=' + (r.tp||'') + ' model=' + (r.model||'')), hoverinfo:'text+x+y'});
    }
  }
  return traces;
}

function renderForKey(key){
  const payload = CLIENT_MAP[key];
  if (!payload){ document.getElementById('plot_div').innerHTML = '<p>No data</p>'; return; }
  const recs = payload.records || [];
  const xcol = xSel.value || 'median_e2el';
  const ycol = ySel.value || '';
  if (!xcol || !ycol){ document.getElementById('plot_div').innerHTML = '<p>Missing X or Y</p>'; return; }
  const connectTp = (tpLine.value==='yes');
  const tpFilter = tpSel.value;
  const precFilter = precSel.value==='Tutte' ? 'all' : precSel.value.toLowerCase();
  const traces = buildTraces(recs, xcol, ycol, connectTp, tpFilter, precFilter);
  const layout = {title: ycol + ' vs ' + xcol, xaxis:{title:xcol}, yaxis:{title:ycol}, legend:{orientation:'v'}};
  Plotly.newPlot('plot_div', traces, layout, {responsive:true});
}

// init
populateModels();
if (modelSel.options.length){
  modelSel.value = modelSel.options[0].value;
  populateContextsForModel(modelSel.value);
  if (ctxSel.options.length){
    ctxSel.value = ctxSel.options[0].value;
    updateRunInfo(ctxSel.value);
    populateYOptionsForKey(ctxSel.value);
    populateTpOptionsForKey(ctxSel.value);
    if (ySel.options.length) ySel.value = ySel.options[0].value;
    renderForKey(ctxSel.value);
  }
}

// events
modelSel.addEventListener('change', ()=>{
  populateContextsForModel(modelSel.value);
  if (ctxSel.options.length){ ctxSel.value = ctxSel.options[0].value; updateRunInfo(ctxSel.value); populateYOptionsForKey(ctxSel.value); populateTpOptionsForKey(ctxSel.value); if (ySel.options.length) ySel.value = ySel.options[0].value; renderForKey(ctxSel.value); }
});
ctxSel.addEventListener('change', ()=>{ updateRunInfo(ctxSel.value); populateYOptionsForKey(ctxSel.value); populateTpOptionsForKey(ctxSel.value); if (ySel.options.length) ySel.value = ySel.options[0].value; renderForKey(ctxSel.value); });
ySel.addEventListener('change', ()=> renderForKey(ctxSel.value));
xSel.addEventListener('change', ()=> renderForKey(ctxSel.value));
precSel.addEventListener('change', ()=> renderForKey(ctxSel.value));
tpSel.addEventListener('change', ()=> renderForKey(ctxSel.value));
tpLine.addEventListener('change', ()=> renderForKey(ctxSel.value));
</script>
"""
    html = header + controls + plot_div + body_js + main_js + "</body></html>"
    return html

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
    html = build_plotly_html(df, data_file_names)
    try:
        OUT_FILE.write_text(html, encoding="utf-8")
        print(f"Wrote {OUT_FILE}")
    except Exception as e:
        print(f"Error: could not write {OUT_FILE}: {e}")

if __name__ == "__main__":
    main()
