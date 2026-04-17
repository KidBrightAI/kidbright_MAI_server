#!/usr/bin/env python
"""Local CLI runner for KidBright MAI training (WSL replacement for Colab).

Calls main.training_task() directly with a StdoutAnnouncer instead of going
through the Flask /upload + /train endpoints.

Usage:
  python run_train.py --project-zip path/to/project.zip
  python run_train.py --project-dir path/to/project_folder
  python run_train.py --project-id existing_id   # uses projects/existing_id/

--project-zip and --project-dir create or refresh projects/{id}/ and
projects/{id}/project.zip so training_task can extract it. --project-id
assumes projects/{id}/ is already prepared.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

os.environ.setdefault("KBMAI_OPENCV_ROOT", os.path.expanduser("~/opencv-3.4.13"))

from utils.stdout_announcer import StdoutAnnouncer  # noqa: E402
import main as _server  # noqa: E402


def _slug(path: str) -> str:
    name = os.path.splitext(os.path.basename(os.path.normpath(path)))[0]
    return name or "project"


def _zip_project_dir(src_dir: str, zip_path: str) -> None:
    """Zip src_dir contents (not src_dir itself) to zip_path, skipping output/ and existing zip."""
    zip_path_abs = os.path.abspath(zip_path)
    if os.path.exists(zip_path_abs):
        os.remove(zip_path_abs)
    with zipfile.ZipFile(zip_path_abs, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(src_dir):
            rel_root = os.path.relpath(root, src_dir)
            if rel_root.startswith("output") or rel_root.startswith("temp") or rel_root.startswith("yolo_dataset"):
                continue
            for f in files:
                abs_f = os.path.join(root, f)
                if os.path.abspath(abs_f) == zip_path_abs:
                    continue
                rel_f = os.path.relpath(abs_f, src_dir)
                zf.write(abs_f, rel_f)


def _prep_from_zip(zip_file: str, project_id: str | None) -> str:
    pid = project_id or _slug(zip_file)
    proj_dir = os.path.join("projects", pid)
    os.makedirs(proj_dir, exist_ok=True)
    dest_zip = os.path.join(proj_dir, "project.zip")
    shutil.copyfile(zip_file, dest_zip)
    return pid


def _prep_from_dir(src_dir: str, project_id: str | None) -> str:
    pid = project_id or _slug(src_dir)
    proj_dir = os.path.join("projects", pid)
    src_abs = os.path.abspath(src_dir)
    proj_abs = os.path.abspath(proj_dir)
    if src_abs != proj_abs:
        if os.path.exists(proj_dir):
            shutil.rmtree(proj_dir)
        shutil.copytree(src_dir, proj_dir)
    zip_path = os.path.join(proj_dir, "project.zip")
    _zip_project_dir(proj_dir, zip_path)
    return pid


def _prep_from_id(pid: str) -> str:
    proj_dir = os.path.join("projects", pid)
    if not os.path.exists(proj_dir):
        raise FileNotFoundError(f"projects/{pid}/ not found")
    zip_path = os.path.join(proj_dir, "project.zip")
    json_path = os.path.join(proj_dir, "project.json")
    if not os.path.exists(zip_path) and not os.path.exists(json_path):
        raise FileNotFoundError(f"Neither project.zip nor project.json in {proj_dir}")
    return pid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--project-zip", help="Path to project.zip exported from IDE")
    grp.add_argument("--project-dir", help="Path to unpacked project folder (contains project.json + dataset/)")
    grp.add_argument("--project-id", help="Existing id under projects/")
    ap.add_argument("--id", dest="override_id", help="Override project_id (useful with --zip/--dir)")
    ap.add_argument("--log-file", help="Mirror stdout to a log file")
    args = ap.parse_args()

    if args.project_zip:
        pid = _prep_from_zip(args.project_zip, args.override_id)
    elif args.project_dir:
        pid = _prep_from_dir(args.project_dir, args.override_id)
    else:
        pid = _prep_from_id(args.project_id)

    q = StdoutAnnouncer(log_file=args.log_file)
    print(f"=== [run_train] project_id={pid}, backend={_server.BACKEND}, device={_server.DEVICE} ===", flush=True)
    _server.training_task(pid, q)
    print(f"=== [run_train] final STAGE={_server.STAGE} (3=trained, <3=failed) ===", flush=True)
    q.close()
    return 0 if _server.STAGE == 3 else 1


if __name__ == "__main__":
    sys.exit(main())
