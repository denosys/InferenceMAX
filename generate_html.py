#!/usr/bin/env python3
# generate_html.py — updated 2025-10-21 (fixed: CDN, lazy-load menu population, client-side model canonicalization sample)
# Changelog (summary):
# - Robust parsing/normalization (hw/hardware, tp, conc, precision)
# - FILE_MAP matching more permissive and model deduplication
# - Client payload reduced for large datasets (lazy load)
# - UI improvements: ordered selects, record counts, export CSV/PNG, log/linear axis toggle
# - diagnostics.txt enriched
# - Responsive CSS + color palette friendly to colorblind users
# Note: keeps backward compatibility with provided sample JSON files.
# - Add _model_canonical and _model_display to CLIENT_MAP (only for plotting/UI)
# - Canonicalization rules applied only when building client payload; original JSON files untouched
# - Ensure model/context selects are populated even when many datasets are lazy-loaded

import json
import os
import math
from pathlib import Path
from typing import List, Any, Optional, Dict
import re
import pandas as pd

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

# Threshold to embed full records client-side; above this we only send schema/columns and lazy-load records.
EMBED_RECORDS_LIMIT = 5000

def list_data_files() -> List[Path]:
    """List JSON files in docs/data (skip directories and hidden files)."""
    if not DATA_DIR.exists():
        return []
    return sorted([p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".json"])

def load_json_safe(p: Path) -> Optional[Any]:
    """Safely load JSON from file, trying different read methods."""
    try:
        return json.loads(p.read_bytes().decode("utf-8"))
    except Exception:
        try:
            return json.load(p.open("r", encoding="utf-8"))
        except Exception:
            return None

def normalize_records_from_json(j: Any) -> List[dict]:
    """Normalize various JSON shapes into a flat list of record dicts."""
    if j is None:
        return []
    if isinstance(j, list):
        return [it for it in j if isinstance(it, dict)]
    if isinstance(j, dict):
        for k in ("results", "data", "records", "items", "files"):
            if k in j and isinstance(j[k], list):
                return [it for it in j[k] if isinstance(it, dict)]
        return [j]
    return []

def coerce_record_types(r: dict) -> dict:
    """Normalize keys and coerce numeric-like strings to numbers for a single record."""
    # normalize hardware key
    if "hw" not in r and "hardware" in r:
        r["hw"] = r.get("hardware")
    if "hw" in r and r["hw"] is not None:
        r["hw"] = str(r["hw"]).lower()
    # precision default
    if "precision" not in r or r["precision"] in (None, ""):
        r["precision"] = "fp8"
    else:
        r["precision"] = str(r["precision"]).lower()
    # try coerce tp, conc to ints if possible
    for k in ("tp", "conc"):
        if k in r and r[k] not in (None, ""):
            try:
                r[k] = int(r[k])
            except Exception:
                try:
                    r[k] = float(r[k])
                except Exception:
                    pass
    # coerce common numeric metrics
    for k in list(r.keys()):
        if k in ("tp", "conc", "hw", "model", "precision", "framework"):
            continue
        v = r.get(k)
        if isinstance(v, str):
            # try numeric
            try:
                if "." in v or "e" in v.lower():
                    r[k] = float(v)
                else:
                    r[k] = int(v)
            except Exception:
                # leave as string
                pass
    return r

# Canonicalization utilities (only used for UI/plotting payload)
_CANONICAL_MAP = {
    # explicit canonical targets
    "llama-3.3-70b-instruct": "Llama-3.3-70B-Instruct",
    "deepseek-r1-0528": "DeepSeek-R1-0528",
    "gpt-oss-120b": "gpt-oss-120b",
}

_DISPLAY_MAP = {
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
    "DeepSeek-R1-0528": "DeepSeek-R1-0528",
    "gpt-oss-120b": "gpt-oss 120B",
}

_suffix_re = re.compile(r"(?:[-_/]?(?:fp8|fp4|mxfp4|mxpf4|kv|preview|v\d+|-v\d+))+$", re.IGNORECASE)
_vendor_prefix_re = re.compile(r"^(?:nvidia/|amd/|deepseek-ai/|openai/|/mnt/.*?/models/)", re.IGNORECASE)
_clean_re = re.compile(r"[_\s]+")

def canonicalize_model_name(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    s = str(raw).strip()
    s_low = s.lower()
    # remove vendor/path prefixes
    s_low = _vendor_prefix_re.sub("", s_low)
    # strip known suffixes like -fp8, -fp4, -kv, -preview, -v2, etc.
    s_low = _suffix_re.sub("", s_low)
    # normalize separators
    s_low = s_low.replace("\\", "-").replace("/", "-")
    s_low = _clean_re.sub("-", s_low)
    s_low = re.sub(r"-{2,}", "-", s_low).strip("-")
    # map known patterns
    if re.search(r"llama.*3.*70b.*instruct", s_low):
        return _CANONICAL_MAP["llama-3.3-70b-instruct"]
    if re.search(r"deepseek.*r1.*0528", s_low):
        return _CANONICAL_MAP["deepseek-r1-0528"]
    if "gpt-oss-120b" in s_low or "gptoss-120b" in s_low:
        return _CANONICAL_MAP["gpt-oss-120b"]
    # fallback: return cleaned candidate
    candidate = s_low if s_low else s
    return candidate

def display_name_for_canonical(canon: str) -> str:
    return _DISPLAY_MAP.get(canon, canon.replace("-", " "))

def build_dataframe_from_files(files: List[Path]) -> pd.DataFrame:
    """Build a normalized pandas DataFrame from json files and infer metadata."""
    recs_all = []
    per_file_counts = {}
    for p in files:
        j = load_json_safe(p)
        if j is None:
            print(f"Warning: failed to parse {p.name}, skipping")
            continue
        recs = normalize_records_from_json(j)
        key = next((k for k in FILE_MAP if k in p.name), None)
        per_file_counts[p.name] = len(recs)
        for r in recs:
            # inject metadata from FILE_MAP when matched
            if key:
                m, isl, osl = FILE_MAP[key]
                r.setdefault("model", m)
                r.setdefault("isl", isl)
                r.setdefault("osl", osl)
                r.setdefault("file_key", key)
            # normalize and coerce types for each record
            r = coerce_record_types(r)
            recs_all.append(r)
    if not recs_all:
        return pd.DataFrame()
    df = pd.json_normalize(recs_all)
    # ensure hw column exists
    if "hw" not in df.columns and "hardware" in df.columns:
        df["hw"] = df["hardware"].astype(str).str.lower()
    if "precision" not in df.columns:
        df["precision"] = "fp8"
    # coerce object columns that are numeric-like
    for c in df.columns:
        if df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().any():
                df[c] = coerced
    return df

def summarize_files(files: List[Path]) -> Dict[str, Any]:
    """Return a summary dict for diagnostics and static summary.json."""
    data_files = [p.name for p in files]
    counts = {}
    for p in files:
        j = load_json_safe(p)
        recs = normalize_records_from_json(j)
        counts[p.name] = len(recs)
    return {"files": data_files, "counts": counts, "total_files": len(files)}

def build_client_payload(files: List[Path]) -> Dict[str, Any]:
    """
    Build a lightweight client-side map:
    - include columns/schema for each dataset
    - embed records only when under EMBED_RECORDS_LIMIT
    - otherwise include record_count and path for lazy load
    - add per-record derived fields (only in payload): _model_canonical, _model_display
    - add 'sample' metadata for lazy entries so client can populate menus
    """
    client_map = {}
    for p in files:
        key = next((k for k in FILE_MAP if k in p.name), None)
        j = load_json_safe(p)
        recs = normalize_records_from_json(j)
        # inject metadata from FILE_MAP when matched (but do not overwrite original model field)
        if key:
            m, isl, osl = FILE_MAP[key]
            for r in recs:
                r.setdefault("model", m); r.setdefault("isl", isl); r.setdefault("osl", osl); r.setdefault("file_key", key)
        # coerce each record minimally; copy to avoid mutating original objects
        recs = [coerce_record_types(dict(r)) for r in recs]
        # derive model canonical/display only in payload
        for r in recs:
            orig = r.get("model") or ""
            canon = canonicalize_model_name(orig)
            r["_model_canonical"] = canon
            r["_model_display"] = display_name_for_canonical(canon)
            r["model_original"] = orig
        cols = list(pd.json_normalize(recs).columns) if recs else []
        entry = {"columns": cols, "record_count": len(recs), "filename": p.name}
        if len(recs) <= EMBED_RECORDS_LIMIT:
            entry["records"] = recs
            # also store a sample for quick client-side inspection
            entry["sample"] = recs[0] if recs else {}
        else:
            entry["records"] = None  # will be lazy-loaded client-side via fetch of /docs/data/<filename>
            # create a sample derived from FILE_MAP or filename so client can populate menus without fetching
            sample = {}
            if key:
                m, isl, osl = FILE_MAP[key]
                canon = canonicalize_model_name(m)
                sample["_model_canonical"] = canon
                sample["_model_display"] = display_name_for_canonical(canon)
                sample["model"] = m
                sample["isl"] = isl
                sample["osl"] = osl
            else:
                # best-effort from filename
                stem = p.stem
                # replace separators for human-readable
                inferred = re.sub(r'[-_.]+', ' ', stem)
                canon = canonicalize_model_name(inferred)
                sample["_model_canonical"] = canon
                sample["_model_display"] = display_name_for_canonical(canon)
                sample["model"] = inferred
            entry["sample"] = sample
        # store under file_key when available, otherwise filename base
        map_key = key or p.stem
        client_map[map_key] = entry
    return client_map

def build_plotly_html(client_map: Dict[str, Any]) -> str:
    """Construct interactive HTML with controls and embedded client_map JSON."""
    header = "<!doctype html><html><head><meta charset='utf-8'><title>InferenceMAX — Interactive</title>"
    # fixed CDN URL
    header += "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"
    header += "<style>"
    header += """
body{font-family:system-ui,Arial,sans-serif;margin:12px;background:#fff;color:#111}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px}
.controls label{font-size:0.9rem;margin-right:6px}
.controls select,input,button{padding:6px 8px;border:1px solid #ccc;border-radius:6px;background:#f8f8f8}
#plot_div{width:100%;height:720px;max-width:1400px;margin-top:8px}
.info_line{font-size:0.9rem;margin-left:8px;color:#333}
.small{font-size:0.85rem;color:#555}
@media(max-width:900px){ .controls{flex-direction:column;align-items:stretch} #plot_div{height:520px} }
"""
    header += "</style></head><body>"
    # controls: model, context, precision, tp, tp_line, x/y, axis scale, export buttons, record count
    controls = (
        "<div class='controls'>"
        "<label>Model:</label><select id='model_sel'></select>"
        "<label>Context (ISL/OSL):</label><select id='ctx_sel'></select>"
        "<label>Precision:</label><select id='prec_sel'><option value='all'>All</option><option value='fp8'>FP8</option><option value='fp4'>FP4</option></select>"
        "<label>TP (cards):</label><select id='tp_sel'><option value='all'>All</option></select>"
        "<label>Connect points:</label><select id='tp_line'><option value='yes'>Yes</option><option value='no'>No</option></select>"
        "<label>X axis:</label><select id='x_sel'><option value='median_e2el'>E2E latency (median)</option><option value='median_intvty'>Interactivity (median)</option><option value='tput_per_gpu'>Throughput per GPU</option></select>"
        "<label>Y axis:</label><select id='y_sel'></select>"
        "<label>Y scale:</label><select id='yscale_sel'><option value='linear'>Linear</option><option value='log'>Log</option></select>"
        "<button id='export_csv'>Export CSV</button>"
        "<button id='export_png'>Export PNG</button>"
        "<button id='reset_view'>Reset view</button>"
        "<span class='info_line small'>Showing <span id='count_shown'>0</span> / <span id='count_total'>0</span> records</span>"
        "</div>"
    )
    plot_div = "<div id='plot_div'></div>"

    # embed client_map
    body_js = f"<script>const CLIENT_MAP = {json.dumps(client_map, ensure_ascii=False)};</script>"

    # main JS
    main_js = """
<script>
const $id=(id)=>document.getElementById(id);
const modelSel=$id('model_sel'), ctxSel=$id('ctx_sel'), precSel=$id('prec_sel'),
tpSel=$id('tp_sel'), tpLine=$id('tp_line'), ySel=$id('y_sel'), xSel=$id('x_sel'),
yscaleSel=$id('yscale_sel'), exportCsv=$id('export_csv'), exportPng=$id('export_png'),
resetView=$id('reset_view'), countShown=$id('count_shown'), countTotal=$id('count_total');

/* Helper canonicalization mirroring server-side heuristics */
function canonicalizeModelFromString(raw){
  if(!raw) return 'unknown';
  let s = String(raw).trim().toLowerCase();
  s = s.replace(/^(nvidia\/|amd\/|deepseek-ai\/|openai\/|\/mnt\/.*?\/models\/)/,'');
  s = s.replace(/(\\-?fp8|\\-?fp4|\\-?mxfp4|\\-?mxpf4|\\-?kv|\\-?preview|\\-v?\\d+)$/i,'');
  s = s.replace(/[\\/,_\\s]+/g,'-').replace(/-+/g,'-').replace(/(^-+|-+$)/g,'');
  if(/llama.*3.*70b.*instruct/.test(s)) return 'Llama-3.3-70B-Instruct';
  if(/deepseek.*r1.*0528/.test(s)) return 'DeepSeek-R1-0528';
  if(s.indexOf('gpt-oss-120b')!==-1 || s.indexOf('gptoss-120b')!==-1) return 'gpt-oss-120b';
  return s || raw;
}

/* Build models map robustly even when many entries are lazy-loaded */
function buildModels(){
  const models = {}; // canon -> {display, contexts:Set}
  for(const key of Object.keys(CLIENT_MAP)){
    const meta = CLIENT_MAP[key];
    const recs = meta.records;
    if(recs && recs.length){
      recs.forEach(r=>{
        const canon = r._model_canonical || (r.model || 'unknown');
        const display = r._model_display || (r.model || canon);
        if(!models[canon]) models[canon] = {display: display, contexts: new Set()};
        models[canon].contexts.add(String(key));
      });
    } else if(meta.sample && (meta.sample._model_canonical || meta.sample.model)){
      const s = meta.sample;
      const canon = s._model_canonical || canonicalizeModelFromString(s.model || '');
      const display = s._model_display || (s.model || canon);
      if(!models[canon]) models[canon] = {display: display, contexts: new Set()};
      models[canon].contexts.add(String(key));
    } else {
      // fallback: infer from filename or key so UI still has entries when everything is lazy
      const fallbackRaw = meta.filename ? meta.filename.replace(/[-_.]+/g,' ') : (key || 'unknown');
      const canon = canonicalizeModelFromString(fallbackRaw);
      const display = canon;
      if(!models[canon]) models[canon] = {display: display, contexts: new Set()};
      models[canon].contexts.add(String(key));
    }
  }
  return models;
}

/* Populate model select from models map; fallback to CLIENT_MAP keys if empty */
function populateModelSelect(){
  modelSel.innerHTML = '';
  const models = buildModels();
  const entries = Object.keys(models).map(k=>({canon:k, display: models[k].display, contexts: models[k].contexts}));
  entries.sort((a,b)=> String(a.display).localeCompare(String(b.display)));
  entries.forEach(e=>{
    modelSel.appendChild(new Option(e.display, e.canon));
  });
  if(!modelSel.options.length){
    // fallback: show filenames/keys
    Object.keys(CLIENT_MAP).forEach(k=>{
      const meta = CLIENT_MAP[k];
      const label = meta.sample && (meta.sample._model_display || meta.sample.model) ? (meta.sample._model_display || meta.sample.model) : (meta.filename || k);
      modelSel.appendChild(new Option(label, k));
    });
  }
}

/* Populate context select (isl/osl) for a canonical model */
function populateContextSelect(modelName){
  ctxSel.innerHTML = '';
  const seen = new Set();
  for(const key of Object.keys(CLIENT_MAP)){
    const meta = CLIENT_MAP[key];
    const recs = meta.records;
    let found = false;
    if(recs && recs.length){
      for(const r of recs){
        const rcanon = r._model_canonical || (r.model || '');
        if(rcanon === modelName){ found = true; break; }
      }
    } else if(meta.sample && (meta.sample._model_canonical || meta.sample.model)){
      const s = meta.sample;
      const canon = s._model_canonical || canonicalizeModelFromString(s.model || '');
      if(canon === modelName) found = true;
    } else if(meta.filename && String(meta.filename).toLowerCase().includes(String(modelName).toLowerCase().replace(/-/g,''))){
      found = true;
    }
    if(found && !seen.has(key)){
      seen.add(key);
      const sample = (meta.records && meta.records.length) ? meta.records[0] : (meta.sample || {});
      const isl = sample.isl || '';
      const osl = sample.osl || '';
      const label = (isl || osl) ? `${isl}/${osl}` : (meta.filename || key);
      ctxSel.appendChild(new Option(label, key));
    }
  }
  // if no contexts found, provide at least entries from CLIENT_MAP (helpful when canonical matching fails)
  if(!ctxSel.options.length){
    Object.keys(CLIENT_MAP).forEach(k=>{
      const meta = CLIENT_MAP[k];
      const label = meta.sample && (meta.sample.isl || meta.sample.osl) ? `${meta.sample.isl||''}/${meta.sample.osl||''}` : (meta.filename || k);
      ctxSel.appendChild(new Option(label, k));
    });
  }
}

/* Lazy-load records when needed; normalize and derive canonical/display client-side */
function loadRecordsIfNeeded(mapKey){
  const meta = CLIENT_MAP[mapKey];
  if(!meta) return Promise.resolve(null);
  if(meta.records !== null && meta.records !== undefined) return Promise.resolve(meta.records);
  if(!meta.filename) return Promise.resolve(null);
  const url = 'data/' + meta.filename;
  return fetch(url).then(resp=>{
    if(!resp.ok) throw new Error('Failed to fetch '+url);
    return resp.json();
  }).then(j=>{
    const recs = (Array.isArray(j)? j : (j.records || j.results || j.data || [j])).filter(r=> typeof r==='object');
    recs.forEach(r=>{
      if(!r.hw && r.hardware) r.hw = r.hardware;
      if(r.hw) r.hw = String(r.hw).toLowerCase();
      if(!r.precision) r.precision = 'fp8';
      ['tp','conc'].forEach(k=>{
        if(r[k]!==undefined && r[k]!==null && r[k]!==''){
          const n = Number(r[k]);
          if(!Number.isNaN(n)) r[k] = n;
        }
      });
      // client-side canonicalization
      const orig = r.model || '';
      let s = String(orig).toLowerCase();
      s = s.replace(/^(nvidia\/|amd\/|deepseek-ai\/|openai\/|\/mnt\/.*?\/models\/)/,'');
      s = s.replace(/(\-?fp8|\-?fp4|\-?mxfp4|\-?mxpf4|\-?kv|\-?preview|\-v?\d+)$/i,'');
      s = s.replace(/[\/_\s]+/g,'-').replace(/-+/g,'-').replace(/(^-+|-+$)/g,'');
      let canon = 'unknown';
      if(/llama.*3.*70b.*instruct/.test(s)) canon = 'Llama-3.3-70B-Instruct';
      else if(/deepseek.*r1.*0528/.test(s)) canon = 'DeepSeek-R1-0528';
      else if(s.indexOf('gpt-oss-120b')!==-1 || s.indexOf('gptoss-120b')!==-1) canon = 'gpt-oss-120b';
      else canon = s || orig;
      r._model_canonical = canon;
      r._model_display = (canon === 'gpt-oss-120b') ? 'gpt-oss 120B' : canon;
      r.model_original = orig;
    });
    meta.records = recs;
    return recs;
  }).catch(err=>{
    console.warn('Lazy load failed:', err);
    return null;
  });
}

/* Populate Y options for a context key based on available columns/samples */
function populateYOptionsForKey(key){
  ySel.innerHTML = '';
  const meta = CLIENT_MAP[key];
  if(!meta) { ySel.appendChild(new Option('(no data)','')); return; }
  let cols = meta.columns || [];
  if(meta.records && meta.records.length){
    cols = Object.keys(meta.records[0]);
  } else if(meta.sample){
    cols = Object.keys(meta.sample);
  }
  const numeric = [];
  cols.forEach(c=>{
    const sample = (meta.records && meta.records[0] && meta.records[0][c]) || (meta.sample && meta.sample[c]) || null;
    if(sample===null || sample===undefined) return;
    if(typeof sample === 'number') numeric.push(c);
    else if(!Number.isNaN(Number(sample))) numeric.push(c);
  });
  const preferred = ['tput_per_gpu','output_tput_per_gpu','median_e2el','median_intvty','median_ttft','p99_e2el','p99_ttft'];
  preferred.forEach(p=>{
    if(numeric.includes(p)) ySel.appendChild(new Option(p,p));
  });
  numeric.filter(c=>!preferred.includes(c)).forEach(c=> ySel.appendChild(new Option(c,c)));
  if(!ySel.options.length) ySel.appendChild(new Option('(no numeric)',''));
}

/* Populate TP options for a context key */
function populateTpOptionsForKey(key){
  tpSel.innerHTML = '';
  const meta = CLIENT_MAP[key];
  const s = new Set();
  if(meta.records && meta.records.length){
    meta.records.forEach(r=>{ if(r.tp!==undefined && r.tp!==null && r.tp!=='' ) s.add(String(r.tp)); });
  } else if(meta.sample && meta.sample.tp!==undefined && meta.sample.tp!==null && meta.sample.tp!==''){
    s.add(String(meta.sample.tp));
  }
  tpSel.appendChild(new Option('All','all'));
  Array.from(s).sort((a,b)=>Number(a)-Number(b)).forEach(v=> tpSel.appendChild(new Option(v,v)));
}

/* Build plotly traces from records with grouping and inline small labels */
function buildTraces(records,xcol,ycol,connectLines,tpFilter,precFilter){
  if(!records || !records.length) return [];
  let recs = records.slice();
  if(precFilter && precFilter!=='all') recs = recs.filter(r=> (String(r.precision||'').toLowerCase())===precFilter);
  if(tpFilter && tpFilter!=='all') recs = recs.filter(r=> String(r.tp)===String(tpFilter));
  const groups = {};
  recs.forEach(r=>{
    const hw = (r.hw||r.hardware||'unknown').toString().toLowerCase();
    const tp = (r.tp===undefined||r.tp==='') ? 'none' : String(r.tp);
    groups[hw] = groups[hw]||{};
    groups[hw][tp] = groups[hw][tp]||[];
    groups[hw][tp].push(r);
  });
  const hwKeys = Object.keys(groups).sort();
  const traces = [];
  const colorPalette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
  let colorIdx = 0;
  const textPositions = ['top center','bottom center','middle left','middle right'];
  hwKeys.forEach(hw=>{
    const tpKeys = Object.keys(groups[hw]).sort((a,b)=>{
      const na=Number(a), nb=Number(b);
      if(!Number.isNaN(na) && !Number.isNaN(nb)) return na-nb;
      return a.localeCompare(b);
    });
    tpKeys.forEach(tp=>{
      const rows = groups[hw][tp];
      if(!rows || !rows.length) return;
      rows.sort((a,b)=>{
        const na=Number(a.conc), nb=Number(b.conc);
        if(!isNaN(na) && !isNaN(nb)) return na-nb;
        return String(a.conc||'').localeCompare(String(b.conc||''));
      });
      const xs = rows.map(r=>{ const v=r[xcol]; const n=Number(v); return Number.isNaN(n)? v : n; });
      const ys = rows.map(r=>{ const v=r[ycol]; const n=Number(v); return Number.isNaN(n)? v : n; });
      const hovertexts = rows.map(r=>{
        const gpu = (r.hw||r.hardware||'unknown');
        const nGPU = (r.tp===undefined||r.tp=='')?'N/A':String(r.tp)+' GPU';
        const conc = (r.conc===undefined||r.conc===null||r.conc=='')?'N/A':String(r.conc);
        const xv = (r[xcol]===undefined||r[xcol]===null)?'':r[xcol];
        const yv = (r[ycol]===undefined||r[ycol]===null)?'':r[ycol];
        return ['GPU: '+gpu,'TP: '+nGPU,'Concurrency: '+conc,'X: '+xv,'Y: '+yv].join('<br>');
      });
      const smallLabels = rows.map(r=> (r.conc!==undefined && r.conc!==null && r.conc!=='') ? String(r.conc) : '');
      const tpos = smallLabels.map((_,i)=> textPositions[i % textPositions.length]);
      const color = colorPalette[colorIdx % colorPalette.length];
      colorIdx++;
      const displayName = (rows[0] && rows[0]._model_display) ? rows[0]._model_display : (rows[0] && rows[0].model) || hw;
      traces.push({
        x: xs,
        y: ys,
        mode: connectLines ? 'lines+markers+text' : 'markers+text',
        name: hw + (tp!=='none' ? ' tp='+tp : ''),
        legendgroup: hw,
        marker: {color: color, size:8},
        line: {shape:'linear', color: color},
        text: smallLabels,
        textposition: tpos,
        textfont: {size:9, color: '#222'},
        hoverinfo: 'text',
        hovertext: hovertexts
      });
    });
  });
  return traces;
}

/* Render plot for a selected context key */
function renderForKey(key){
  const meta = CLIENT_MAP[key];
  if(!meta){ $id('plot_div').innerHTML = '<p>No data</p>'; return; }
  loadRecordsIfNeeded(key).then(records=>{
    const recs = meta.records || records || [];
    countTotal.textContent = meta.record_count || recs.length || 0;
    const xcol = xSel.value || 'median_e2el';
    const ycol = ySel.value || '';
    if(!xcol || !ycol){ $id('plot_div').innerHTML = '<p>Missing X or Y</p>'; return; }
    const traces = buildTraces(recs, xcol, ycol, tpLine.value==='yes', tpSel.value, (precSel.value||'').toString().toLowerCase());
    countShown.textContent = traces.reduce((s,t)=> s + (t.x? t.x.length : 0), 0);
    const layout = {
      title: ycol + ' vs ' + xcol,
      xaxis: {title: xcol, type: 'linear'},
      yaxis: {title: ycol, type: yscaleSel.value || 'linear'},
      legend: {orientation: 'v'},
      hovermode: 'closest',
      margin: {t:50, r:20, l:60, b:60}
    };
    Plotly.newPlot('plot_div', traces, layout, {responsive:true});
  });
}

/* Initialize UI and wire events */
populateModelSelect();
if(modelSel.options.length){
  modelSel.value = modelSel.options[0].value;
  populateContextSelect(modelSel.value);
  if(ctxSel.options.length){
    ctxSel.value = ctxSel.options[0].value;
    populateYOptionsForKey(ctxSel.value);
    populateTpOptionsForKey(ctxSel.value);
    if(ySel.options.length) ySel.value = ySel.options[0].value;
    renderForKey(ctxSel.value);
  }
}

modelSel.addEventListener('change', ()=>{
  populateContextSelect(modelSel.value);
  if(ctxSel.options.length){
    ctxSel.value = ctxSel.options[0].value;
    populateYOptionsForKey(ctxSel.value);
    populateTpOptionsForKey(ctxSel.value);
    if(ySel.options.length) ySel.value = ySel.options[0].value;
    renderForKey(ctxSel.value);
  }
});
ctxSel.addEventListener('change', ()=>{
  populateYOptionsForKey(ctxSel.value);
  populateTpOptionsForKey(ctxSel.value);
  if(ySel.options.length) ySel.value = ySel.options[0].value;
  renderForKey(ctxSel.value);
});
['change'].forEach(ev=>{
  ySel.addEventListener(ev, ()=> renderForKey(ctxSel.value));
  xSel.addEventListener(ev, ()=> renderForKey(ctxSel.value));
  precSel.addEventListener(ev, ()=> renderForKey(ctxSel.value));
  tpSel.addEventListener(ev, ()=> renderForKey(ctxSel.value));
  tpLine.addEventListener(ev, ()=> renderForKey(ctxSel.value));
  yscaleSel.addEventListener(ev, ()=> renderForKey(ctxSel.value));
});

/* Export CSV of visible traces */
exportCsv.addEventListener('click', ()=>{
  const gd = document.getElementById('plot_div');
  const data = gd.data || [];
  if(!data.length){ alert('No data to export'); return; }
  let rows = [];
  data.forEach(trace=>{
    const xs = trace.x || [];
    const ys = trace.y || [];
    for(let i=0;i<xs.length;i++){
      rows.push([trace.name, xs[i], ys[i]]);
    }
  });
  const csv = ['trace,x,y'].concat(rows.map(r=> r.map(v=> '"'+String(v).replace(/"/g,'""')+'"').join(','))).join('\\n');
  const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'plot_export.csv'; a.click();
  URL.revokeObjectURL(url);
});

/* Export PNG via Plotly */
exportPng.addEventListener('click', ()=>{
  Plotly.toImage('plot_div',{format:'png',height:800,width:1200}).then(dataUrl=>{
    const a = document.createElement('a'); a.href = dataUrl; a.download = 'plot_snapshot.png'; a.click();
  });
});

/* Reset view */
resetView.addEventListener('click', ()=> { Plotly.relayout('plot_div', { 'xaxis.autorange': true, 'yaxis.autorange': true }); });

/* Initial counts */
countTotal.textContent = 0;
countShown.textContent = 0;

</script>
"""

    return header + controls + plot_div + body_js + main_js + "</body></html>"

def write_diagnostics(files: List[Path], df: pd.DataFrame):
    """Write diagnostics with per-file summaries and dataframe sample info."""
    try:
        with DIAG_FILE.open("w", encoding="utf-8") as d:
            d.write("Diagnostics generated on 2025-10-21\n")
            d.write(f"data_dir: {DATA_DIR}\n\n")
            for p in files:
                j = load_json_safe(p)
                recs = normalize_records_from_json(j)
                d.write(f"file: {p.name} - records: {len(recs)}\n")
                # list common issues: missing hw/precision/tp/conc
                missing = []
                for k in ('hw','precision','tp','conc'):
                    if any((k not in r or r[k] in (None,'') ) for r in recs):
                        missing.append(k)
                if missing:
                    d.write(f"  missing_keys_in_some_records: {missing}\n")
            d.write("\\nDataFrame sample:\\n")
            if df is None or df.empty:
                d.write("  (no records)\\n")
            else:
                d.write(json.dumps(json.loads(df.head(3).to_json(orient='records')), indent=2, ensure_ascii=False))
        print(f"Wrote diagnostics: {DIAG_FILE}")
    except Exception as e:
        print("Warning: could not write diagnostics:", e)

def main():
    # discover files
    data_files = list_data_files()
    print("INFO: discovered data files:", [p.name for p in data_files])

    # build dataframe for diagnostics (but we will send lightweight client_map)
    df = build_dataframe_from_files(data_files)
    records_count = 0 if df.empty else len(df)
    sample_record = df.iloc[0].to_dict() if records_count>0 else None

    # write diagnostics
    write_diagnostics(data_files, df)

    # write static summary
    try:
        summary = summarize_files(data_files)
        (STATIC_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print("Wrote static summary")
    except Exception as e:
        print("Warning: could not write static summary:", e)

    # build client payload and html
    client_map = build_client_payload(data_files)
    html = build_plotly_html(client_map)
    try:
        OUT_FILE.write_text(html, encoding='utf-8')
        print(f"Wrote {OUT_FILE}")
    except Exception as e:
        print("Error: could not write index.html:", e)

if __name__ == '__main__':
    main()
