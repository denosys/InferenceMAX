#!/usr/bin/env python3
# scripts/compact_schema_all_examples.py
# Python 3.7+ compatible
import sys
import os
import json
from collections import OrderedDict

def detect_type(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int) and not isinstance(v, bool):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "unknown"

def add_example(store, key, val):
    entry = store.setdefault(key, {"type": detect_type(val), "examples": [], "numeric_stats": None})
    t = detect_type(val)
    if entry["type"] != t and entry["type"] != "mixed":
        entry["type"] = "mixed"
    # keep examples for strings and booleans
    if isinstance(val, str) or isinstance(val, bool):
        if val not in entry["examples"]:
            entry["examples"].append(val)
    elif isinstance(val, list):
        sig = {"len": len(val)}
        if sig not in entry["examples"]:
            entry["examples"].append(sig)
    elif isinstance(val, dict):
        sig = {"keys": sorted(list(val.keys()))}
        if sig not in entry["examples"]:
            entry["examples"].append(sig)
    # numeric stats for ints/floats
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if entry.get("numeric_stats") is None:
            entry["numeric_stats"] = {"min": val, "max": val}
        else:
            ns = entry["numeric_stats"]
            if val < ns["min"]:
                ns["min"] = val
            if val > ns["max"]:
                ns["max"] = val

def analyze_item(store, obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            add_example(store, k, v)

def compact_schema_from_items(items):
    store = OrderedDict()
    for it in items:
        analyze_item(store, it)
    schema = OrderedDict()
    for k, v in store.items():
        entry = OrderedDict()
        entry["type"] = v["type"]
        if v["examples"]:
            entry["examples"] = v["examples"]
        if v.get("numeric_stats") is not None:
            entry["numeric_stats"] = v["numeric_stats"]
        schema[k] = entry
    return schema

def schema_signature(schema):
    # deterministic signature for schema: tuple of (key,type) sorted by key and include numeric_stats keys presence
    parts = []
    for k in sorted(schema.keys()):
        e = schema[k]
        t = e.get("type","")
        ns = e.get("numeric_stats")
        examples = e.get("examples")
        part = (k, t, bool(ns), tuple(examples) if examples is not None else None)
        parts.append(part)
    return tuple(parts)

def process_file(inpath, outpath):
    with open(inpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]
    schema = compact_schema_from_items(items)
    outdata = {"schema": schema}
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(outdata, f, ensure_ascii=False, indent=2)
    return schema

def main():
    if len(sys.argv) != 3:
        print("Usage: compact_schema_all_examples.py <input_dir> <output_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.isdir(input_dir):
        print("Input directory not found:", input_dir)
        sys.exit(2)
    os.makedirs(output_dir, exist_ok=True)

    schema_pool = OrderedDict()   # id -> schema
    sig_to_id = {}
    files_list = []

    for fn in sorted(os.listdir(input_dir)):
        if not fn.lower().endswith(".json"):
            continue
        inpath = os.path.join(input_dir, fn)
        outpath = os.path.join(output_dir, fn)
        try:
            schema = process_file(inpath, outpath)
            sig = schema_signature(schema)
            if sig in sig_to_id:
                sid = sig_to_id[sig]
            else:
                sid = "s{}".format(len(schema_pool)+1)
                sig_to_id[sig] = sid
                schema_pool[sid] = schema
            files_list.append({
                "path": inpath.replace("\\", "/"),
                "filename": fn,
                "schema_id": sid
            })
            print("Processed:", fn, "-> schema_id", sid)
        except Exception as e:
            print("Error processing", fn, ":", e)

    summary = {"schema_pool": schema_pool, "files": files_list}
    summary_path = os.path.join(output_dir, "summary_compact.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Wrote summary_compact.json to:", summary_path)

if __name__ == "__main__":
    main()
