#!/usr/bin/env python3
# generate_html.py (compact, comments preserved)
import json, os
from pathlib import Path
from typing import List, Any, Optional
import pandas as pd
import plotly.graph_objects as go

# root and output directories
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "docs"
DATA_DIR = OUT_DIR / "data"
OUT_FILE = OUT_DIR / "index.html"
DIAG_FILE = OUT_DIR / "diagnostics.txt"
STATIC_DIR = OUT_DIR / "static"
TMP_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "generate_html_tmp"

# ensure directories exist
for d in (OUT_DIR, STATIC_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

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
    # List JSON files in docs/data (skip older subdir)
    if not DATA_DIR.exists(): return []
    return sorted([p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".json" and p != DATA_DIR / "older"])

def load_json_safe(p: Path) -> Optional[Any]:
    # Safely load JSON from file (try raw bytes then file open)
    try:
        return json.loads(p.read_bytes().decode("utf-8"))
    except Exception:
        try:
            return json.load(p.open("r", encoding="utf-8"))
        except Exception:
            return None

def normalize_records_from_json(j: Any) -> List[dict]:
    # Normalize JSON payloads into a list of record dicts
    if j is None: return []
    if isinstance(j, list): return [it for it in j if isinstance(it, dict)]
    if isinstance(j, dict):
        for k in ("results","data","records","items"):
            if k in j and isinstance(j[k], list):
                return [it for it in j[k] if isinstance(it, dict)]
        return [j]
    return []

def build_dataframe_from_files(files: List[Path]) -> pd.DataFrame:
    # Build a normalized pandas DataFrame from json files with inferred metadata
    recs_all=[]
    for p in files:
        j = load_json_safe(p)
        if j is None:
            print(f"Warning: failed to parse {p.name}, skipping"); continue
        recs = normalize_records_from_json(j)
        key = next((k for k in FILE_MAP if k in p.name), None)
        for r in recs:
            if key:
                m, isl, osl = FILE_MAP[key]
                r.setdefault("model", m); r.setdefault("isl", isl); r.setdefault("osl", osl); r.setdefault("file_key", key)
        recs_all.extend(recs)
    if not recs_all: return pd.DataFrame()
    df = pd.json_normalize(recs_all)
    # normalize hw name and ensure precision column
    if "hw" in df.columns: df["hw"]=df["hw"].astype(str).str.lower()
    elif "hardware" in df.columns: df["hw"]=df["hardware"].astype(str).str.lower()
    if "precision" not in df.columns: df["precision"]="fp8"
    # coerce numeric-like strings to numeric where appropriate
    for c in df.columns:
        if df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().any(): df[c]=coerced
    return df

def build_plotly_html(df: pd.DataFrame, data_file_names: List[str]) -> str:
    # Construct client-side map of available datasets (only keys present)
    client_map={}
    for key,(m,isl,osl) in FILE_MAP.items():
        match = next((n for n in data_file_names if key in n), None)
        if not match: continue
        recs = normalize_records_from_json(load_json_safe(DATA_DIR / match) or [])
        for r in recs:
            r.setdefault("model", m); r.setdefault("isl", isl); r.setdefault("osl", osl)
            r.setdefault("conc", r.get("conc", None))
            if r.get("conc","") not in (None,""):
                try: r["conc"]=int(r["conc"])
                except Exception:
                    try: r["conc"]=float(r["conc"])
                    except Exception: pass
        client_map[key] = {"columns": list(pd.json_normalize(recs).columns) if recs else [], "records": recs}

    # Build HTML with controls and embedded client data
    header = "<!doctype html><html><head><meta charset='utf-8'><title>InferenceMAX — Interactive</title>"
    header += "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"
    header += "<style>body{font-family:system-ui,Arial,sans-serif;margin:12px;} select{margin-right:8px;} .controls{margin-bottom:8px;}</style></head><body>"
    controls = ("<div class='controls'>"
                "<label>Modello:</label><select id='model_sel'></select>"
                "<label>ISL/OSL:</label><select id='ctx_sel'></select>"
                "<label>Precisione:</label><select id='prec_sel'><option value='all'>Tutte</option><option value='fp8'>FP8</option><option value='fp4'>FP4</option></select>"
                "<label>TP (cards):</label><select id='tp_sel'><option value='all'>All</option></select>"
                "<label>Connetti TP:</label><select id='tp_line'><option value='yes'>Sì</option><option value='no' selected>No</option></select>"
                "<label>Asse Y:</label><select id='y_sel'></select>"
                "<label>Asse X:</label><select id='x_sel'><option value='median_e2el'>E2E latency</option><option value='median_intvty'>Interactivity</option></select>"
                "<label>Run/Artifact:</label><strong id='run_info'></strong>"
                "</div>")
    plot_div = "<div id='plot_div' style='width:100%;height:700px;'></div>"
    body_js = f"<script>const CLIENT_MAP = {json.dumps(client_map)}; const FILE_MAP = {json.dumps(FILE_MAP)};</script>"

    # Main JS: populate controls and render plot (kept compact)
    main_js = """
<script>
const $id=(id)=>document.getElementById(id);
const modelSel=$id('model_sel'),ctxSel=$id('ctx_sel'),precSel=$id('prec_sel'),tpSel=$id('tp_sel'),
tpLine=$id('tp_line'),ySel=$id('y_sel'),xSel=$id('x_sel'),runInfo=$id('run_info');

// Build models map from CLIENT_MAP
function buildModels(){ const m={}; for(const k of Object.keys(CLIENT_MAP)){ const recs=CLIENT_MAP[k].records||[]; if(!recs.length) continue; const r=recs[0]; const name=r.model||'unknown'; m[name]=m[name]||[]; m[name].push([r.isl||'', r.osl||'', k]); } return m; }
const MODELS=buildModels();

// Populate model and context selects
function populateModels(){ modelSel.innerHTML=''; for(const m of Object.keys(MODELS)) modelSel.appendChild(new Option(m,m)); }
function populateContexts(m){ ctxSel.innerHTML=''; (MODELS[m]||[]).forEach(e=>ctxSel.appendChild(new Option(e[0]+'/'+e[1], e[2]))); }
function updateRunInfo(k){ runInfo.textContent=k||''; }

// Populate Y options by scanning first record for numeric columns
function populateYOptionsForKey(key){
  ySel.innerHTML=''; const payload=CLIENT_MAP[key]; if(!payload){ ySel.appendChild(new Option('(no data)','')); return; }
  const cols=payload.columns||[]; const rec=(payload.records && payload.records[0])||{};
  const numeric = cols.filter(c=>{ const v=rec[c]; return v!==undefined&&v!==null&&v!==''&& !isNaN(Number(v)); });
  if(numeric.includes('tput_per_gpu')) ySel.appendChild(new Option('tput_per_gpu','tput_per_gpu'));
  numeric.filter(c=>c!=='tput_per_gpu').forEach(c=>ySel.appendChild(new Option(c,c)));
  if(!ySel.options.length) ySel.appendChild(new Option('(no numeric)',''));
}

// Populate TP (cards) options
function populateTpOptionsForKey(key){
  tpSel.innerHTML=''; const payload=CLIENT_MAP[key]; const s=new Set();
  (payload.records||[]).forEach(r=>{ const v=r['tp']; if(v!==undefined&&v!==null&&v!=='') s.add(String(v));});
  tpSel.appendChild(new Option('All','all')); Array.from(s).sort((a,b)=>Number(a)-Number(b)).forEach(v=>tpSel.appendChild(new Option(v,v)));
}

// Build Plotly traces grouped by hardware and tp
function buildTraces(records,xcol,ycol,connectLines,tpFilter,precFilter){
  if(!records||!records.length) return [];
  let recs=records.slice(); if(precFilter&&precFilter!=='all') recs=recs.filter(r=> (r.precision||'').toString().toLowerCase()===precFilter);
  if(tpFilter&&tpFilter!=='all') recs=recs.filter(r=> String(r.tp)===String(tpFilter));
  const groups={};
  recs.forEach(r=>{
    const hw=(r.hw||r.hardware||'unknown').toString().toLowerCase();
    const tp=(r.tp===undefined||r.tp==='')?'none':String(r.tp);
    groups[hw]=groups[hw]||{}; groups[hw][tp]=groups[hw][tp]||[]; groups[hw][tp].push(r);
  });
  const hwKeys=Object.keys(groups).sort(); const traces=[];
  hwKeys.forEach(hw=> Object.keys(groups[hw]).sort((a,b)=>{const na=Number(a),nb=Number(b); if(!isNaN(na)&&!isNaN(nb)) return na-nb; return a.localeCompare(b);}).
    forEach(tp=>{
      let rows=groups[hw][tp].filter(r=> r.conc!==undefined && r.conc!==null && r.conc!=='');
      if(!rows.length) return;
      rows.sort((a,b)=>{ const na=Number(a.conc), nb=Number(b.conc); if(!isNaN(na)&&!isNaN(nb)) return na-nb; return String(a.conc).localeCompare(String(b.conc));});
      const xs=rows.map(r=>{ const v=r[xcol]; if(v===undefined||v===null||v==='') return null; const n=Number(v); return isNaN(n)? v : n;});
      const ys=rows.map(r=>{ const v=r[ycol]; if(v===undefined||v===null||v==='') return null; const n=Number(v); return isNaN(n)? v : n;});
      const texts=rows.map(r=>{ const gpu=(r.hw||r.hardware||'unknown'); const nGPU=(r.tp===undefined||r.tp=='')?'N/A':String(r.tp)+' GPU'; const conc=(r.conc===undefined||r.conc===null||r.conc=='')?'N/A':String(r.conc)+' Users'; const xv=(r[xcol]===undefined||r[xcol]===null)?'':r[xcol]; const yv=(r[ycol]===undefined||r[ycol]===null)?'':r[ycol]; return ['GPU: '+gpu,'N GPU: '+nGPU,'Concurrency request: '+conc,'X: '+xv,'Y: '+yv].join('<br>');});
      traces.push({x:xs,y:ys,mode:connectLines?'lines+markers':'markers',name:hw+(tp!=='none'?' tp='+tp:''),legendgroup:hw,text:texts,hoverinfo:'text+x+y'});
    }));
  return traces;
}

// Render plot for selected key
function renderForKey(key){
  const payload=CLIENT_MAP[key];
  if(!payload){ document.getElementById('plot_div').innerHTML='<p>No data</p>'; return; }
  const recs=payload.records||[]; const xcol=xSel.value||'median_e2el'; const ycol=ySel.value||'';
  if(!xcol||!ycol){ document.getElementById('plot_div').innerHTML='<p>Missing X or Y</p>'; return; }
  const traces=buildTraces(recs,xcol,ycol,tpLine.value==='yes', tpSel.value, (precSel.value||'').toString().toLowerCase()==='tutte'?'all':(precSel.value||'').toString().toLowerCase());
  Plotly.newPlot('plot_div', traces, {title:ycol+' vs '+xcol, xaxis:{title:xcol}, yaxis:{title:ycol}, legend:{orientation:'v'}}, {responsive:true});
}

// Initialize controls and render first available dataset
populateModels();
if(modelSel.options.length){ modelSel.value=modelSel.options[0].value; populateContexts(modelSel.value); if(ctxSel.options.length){ ctxSel.value=ctxSel.options[0].value; updateRunInfo(ctxSel.value); populateYOptionsForKey(ctxSel.value); populateTpOptionsForKey(ctxSel.value); if(ySel.options.length) ySel.value=ySel.options[0].value; renderForKey(ctxSel.value); }}

// Wire events to update plot on change
modelSel.addEventListener('change', ()=>{ populateContexts(modelSel.value); if(ctxSel.options.length){ ctxSel.value=ctxSel.options[0].value; updateRunInfo(ctxSel.value); populateYOptionsForKey(ctxSel.value); populateTpOptionsForKey(ctxSel.value); if(ySel.options.length) ySel.value=ySel.options[0].value; renderForKey(ctxSel.value);} });
ctxSel.addEventListener('change', ()=>{ updateRunInfo(ctxSel.value); populateYOptionsForKey(ctxSel.value); populateTpOptionsForKey(ctxSel.value); if(ySel.options.length) ySel.value=ySel.options[0].value; renderForKey(ctxSel.value); });
['change'].forEach(ev=>{ ySel.addEventListener(ev, ()=> renderForKey(ctxSel.value)); xSel.addEventListener(ev, ()=> renderForKey(ctxSel.value)); precSel.addEventListener(ev, ()=> renderForKey(ctxSel.value)); tpSel.addEventListener(ev, ()=> renderForKey(ctxSel.value)); tpLine.addEventListener(ev, ()=> renderForKey(ctxSel.value)); });
</script>
"""
    return header + controls + plot_div + body_js + main_js + "</body></html>"

def write_diagnostics(zip_files: List[str], files_in_data: List[str], records_count: int, sample_record: Optional[dict]):
    # Write diagnostics file with discovered zips, data files and a sample record
    try:
        with DIAG_FILE.open("w", encoding="utf-8") as d:
            d.write(f"zip_files_present_in_repo_dir (not used): {zip_files}\n")
            d.write(f"files_in_docs_data (read-only): {files_in_data}\n")
            d.write(f"records_count: {records_count}\nsample_record:\n")
            d.write(json.dumps(sample_record, indent=2) if sample_record else "NONE\n")
        print(f"Wrote diagnostics: {DIAG_FILE}")
    except Exception as e:
        print(f"Warning: could not write diagnostics: {e}")

def main():
    # Discover zip files and data files
    zip_paths = sorted(p.name for p in ROOT.glob("data_zips/*.zip"))
    data_files = list_data_files(); data_file_names = [p.name for p in data_files]
    print(f"INFO: data files discovered (read-only): {data_file_names}")

    # Build DataFrame and sample/record counts
    df = build_dataframe_from_files(data_files)
    records_count = 0 if df.empty else len(df)
    sample_record = df.iloc[0].to_dict() if records_count>0 else None

    # Write diagnostics
    write_diagnostics(zip_paths, data_file_names, records_count, sample_record)

    # Write static summary.json into docs/static
    try:
        summary = {"files": data_file_names, "records_count": records_count}
        (STATIC_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote static summary: {STATIC_DIR/'summary.json'}")
    except Exception as e:
        print(f"Warning: could not write static summary: {e}")

    # Build and write interactive HTML
    html = build_plotly_html(df, data_file_names)
    try:
        OUT_FILE.write_text(html, encoding="utf-8"); print(f"Wrote {OUT_FILE}")
    except Exception as e:
        print(f"Error: could not write {OUT_FILE}: {e}")

if __name__ == "__main__":
    main()
