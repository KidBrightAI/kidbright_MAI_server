#!/usr/bin/env python
"""Local CLI runner for KidBright MAI model conversion (ONNX -> NCNN INT8).

Calls main.convert_model() directly with a StdoutAnnouncer. For V831 (kidbright-mai)
boards this produces model_int8.bin + model_int8.param in projects/{id}/output/.

Usage:
  python run_convert.py --project-id my_project
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

os.environ.setdefault("KBMAI_OPENCV_ROOT", os.path.expanduser("~/opencv-3.4.13"))

from utils.stdout_announcer import StdoutAnnouncer  # noqa: E402
import main as _server  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-id", required=True, help="Existing project id under projects/")
    ap.add_argument("--log-file", help="Mirror stdout to a log file")
    args = ap.parse_args()

    proj_dir = os.path.join("projects", args.project_id)
    if not os.path.exists(os.path.join(proj_dir, "project.json")):
        print(f"ERROR: projects/{args.project_id}/project.json not found — run training first", file=sys.stderr)
        return 2

    q = StdoutAnnouncer(log_file=args.log_file)
    print(f"=== [run_convert] project_id={args.project_id} ===", flush=True)
    _server.convert_model(args.project_id, q)
    print(f"=== [run_convert] final STAGE={_server.STAGE} (5=converted) ===", flush=True)
    q.close()
    return 0 if _server.STAGE == 5 else 1


if __name__ == "__main__":
    sys.exit(main())
