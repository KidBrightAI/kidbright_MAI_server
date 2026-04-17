#!/usr/bin/env python
"""Push a converted model to the V831 mAI board over ADB.

Computes MD5 of model_int8.bin (same hash scheme as IDE's web-adb uploader)
and pushes both .bin + .param to /root/model/{hash}.{ext} on the board.

Usage:
  python adb_deploy.py --project-id my_project
  python adb_deploy.py --project-id my_project --serial 20080411
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def adb(args: list[str], serial: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += args
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def list_devices() -> list[str]:
    out = adb(["devices"], check=False).stdout
    serials = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if line and "\tdevice" in line:
            serials.append(line.split("\t", 1)[0])
    return serials


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--serial", help="ADB serial (default: only connected device)")
    ap.add_argument("--bin", default="model_int8.bin", help="Filename in output/ to treat as .bin")
    ap.add_argument("--param", default="model_int8.param", help="Filename in output/ to treat as .param")
    ap.add_argument("--target-dir", default="/root/model", help="Board dir to push into")
    args = ap.parse_args()

    out_dir = os.path.join("projects", args.project_id, "output")
    bin_path = os.path.join(out_dir, args.bin)
    param_path = os.path.join(out_dir, args.param)
    if not os.path.exists(bin_path):
        print(f"ERROR: {bin_path} not found — run run_convert.py first", file=sys.stderr)
        return 2
    if not os.path.exists(param_path):
        print(f"ERROR: {param_path} not found", file=sys.stderr)
        return 2

    devices = list_devices()
    if not devices:
        print("ERROR: no ADB device detected. Is `adb devices` empty?", file=sys.stderr)
        return 3
    serial = args.serial
    if not serial:
        if len(devices) > 1:
            print(f"ERROR: multiple devices found ({devices}), pass --serial", file=sys.stderr)
            return 3
        serial = devices[0]

    model_hash = md5_of(bin_path)
    print(f"[adb_deploy] serial={serial}  hash={model_hash}")
    print(f"[adb_deploy] bin   = {bin_path} ({os.path.getsize(bin_path)} B)")
    print(f"[adb_deploy] param = {param_path} ({os.path.getsize(param_path)} B)")

    adb(["shell", f"mkdir -p {args.target_dir}"], serial=serial)

    target_bin = f"{args.target_dir}/{model_hash}.bin"
    target_param = f"{args.target_dir}/{model_hash}.param"
    print(f"[adb_deploy] pushing → {target_bin}")
    subprocess.run(["adb", "-s", serial, "push", bin_path, target_bin], check=True)
    print(f"[adb_deploy] pushing → {target_param}")
    subprocess.run(["adb", "-s", serial, "push", param_path, target_param], check=True)

    # verify
    out = adb(["shell", f"ls -la {args.target_dir}/{model_hash}.*"], serial=serial).stdout
    print(out.strip())
    print(f"[adb_deploy] OK — model.hash = {model_hash}")
    print(f"[adb_deploy] Reference on board: /root/model/{model_hash}.bin, /root/model/{model_hash}.param")
    return 0


if __name__ == "__main__":
    sys.exit(main())
