#!/usr/bin/env python3
# compact_schema_all_examples_with_summary.py
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
    entry = store.setdefault(key, {"type": detect_type(val), "examples": []})
    t = detect_type(val)
    if entry["type"] != t:
        entry["type"] = "mixed"
    # keep examples for non-numeric types (strings, bools, arrays, objects)
    if isinstance(val, (str, bool)):
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
    # do not store int/float example values (only keep type)

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
        schema[k] = entry
    return {"schema": schema}

def process_file(inpath, outpath):
    with open(inpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]
    compact = compact_schema_from_items(items)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)
    return compact

def main():
    if len(sys.argv) != 3:
        print("Usage: compact_schema_all_examples_with_summary.py <input_dir> <output_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.isdir(input_dir):
        print("Input directory not found:", input_dir)
        sys.exit(2)
    os.makedirs(output_dir, exist_ok=True)

    summary = {"files": []}

    for fn in sorted(os.listdir(input_dir)):
        if not fn.lower().endswith(".json"):
            continue
        inpath = os.path.join(input_dir, fn)
        outpath = os.path.join(output_dir, fn)
        try:
            compact = process_file(inpath, outpath)
            summary["files"].append({
                "path": inpath.replace("\\", "/"),
                "filename": fn,
                "schema": compact.get("schema", {})
            })
            print("Processed:", fn)
        except Exception as e:
            print("Error processing", fn, ":", e)

    # write summary file
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Summary written to:", summary_path)

if __name__ == "__main__":
    main()
