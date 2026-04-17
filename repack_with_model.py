#!/usr/bin/env python
"""Re-pack an IDE project zip so it carries the trained+converted model.

Mirrors ProjectIOService.saveProject() in the IDE frontend:
  - project.json:  set `model` field to { name, type, hash }
  - zip layout:    add  model/model.bin  and  model/model.param  (or .cvimodel + .mud)

Hash scheme: md5 of the .bin file. Matches web-adb upload + kidbright-mai
board runtime which loads /root/model/{hash}.{bin,param}.

Usage (V831 / kidbright-mai):
  python repack_with_model.py --project-id dog_cat_classify \\
      --src-zip dog_cat_classify.zip \\
      --out-zip dog_cat_classify_trained.zip
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-id", required=True, help="Existing id under projects/ (where output/ lives)")
    ap.add_argument("--src-zip", required=True, help="Original project.zip from the IDE")
    ap.add_argument("--out-zip", required=True, help="Output path for the repacked zip")
    ap.add_argument("--bin", default="model_int8.bin", help="Filename in output/ to treat as model.bin")
    ap.add_argument("--param", default="model_int8.param", help="Filename in output/ to treat as model.param")
    ap.add_argument("--board", default="kidbright-mai", choices=["kidbright-mai", "kidbright-mai-plus"],
                    help="Target board (determines bin/param vs cvimodel/mud)")
    args = ap.parse_args()

    out_dir = os.path.join("projects", args.project_id, "output")
    bin_path = os.path.join(out_dir, args.bin)
    param_path = os.path.join(out_dir, args.param)
    if not os.path.exists(bin_path):
        print(f"ERROR: {bin_path} missing — run run_convert.py first", file=sys.stderr)
        return 2
    if not os.path.exists(param_path):
        print(f"ERROR: {param_path} missing", file=sys.stderr)
        return 2
    if not os.path.exists(args.src_zip):
        print(f"ERROR: source zip {args.src_zip} not found", file=sys.stderr)
        return 2

    bin_data = read_file(bin_path)
    param_data = read_file(param_path)
    model_hash = md5_bytes(bin_data)

    if args.board == "kidbright-mai-plus":
        type_key, ext2 = "cvimodel", "mud"
    else:
        type_key, ext2 = "bin", "param"

    model_meta = {
        "name": "model",
        "type": type_key,
        "hash": model_hash,
    }

    print(f"[repack] project_id = {args.project_id}")
    print(f"[repack] bin   = {bin_path} ({len(bin_data)} B) md5={model_hash}")
    print(f"[repack] param = {param_path} ({len(param_data)} B)")
    print(f"[repack] model = {model_meta}")

    with zipfile.ZipFile(args.src_zip, "r") as zin:
        names = zin.namelist()
        project_json_raw = zin.read("project.json").decode("utf-8")
        project = json.loads(project_json_raw)

        # project.json invariants: carry over everything, only set model.
        project["model"] = model_meta
        # Bonus: bump lastUpdate so IDE knows state changed
        import time as _t
        project["lastUpdate"] = int(_t.time() * 1000)

        new_project_json = json.dumps(project)

        os.makedirs(os.path.dirname(os.path.abspath(args.out_zip)) or ".", exist_ok=True)
        with zipfile.ZipFile(args.out_zip, "w", zipfile.ZIP_STORED) as zout:  # STORE to match IDE saveProject
            for name in names:
                if name == "project.json":
                    zout.writestr("project.json", new_project_json)
                elif name.startswith("model/"):
                    # skip any pre-existing model folder — we'll re-write below
                    continue
                else:
                    zout.writestr(name, zin.read(name))
            # Add model folder
            zout.writestr(f"model/model.{type_key}", bin_data)
            zout.writestr(f"model/model.{ext2}", param_data)

    print(f"[repack] wrote {args.out_zip} ({os.path.getsize(args.out_zip)} B)")
    print(f"[repack] IDE will see workspaceStore.model = {model_meta}")
    print(f"[repack] Board file path after upload: /root/model/{model_hash}.{type_key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
