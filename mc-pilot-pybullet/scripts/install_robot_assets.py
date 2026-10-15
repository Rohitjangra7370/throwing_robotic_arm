#!/usr/bin/env python3
"""Install the vendored Kinova Gen3 model into the active `pybullet_data` package.

Every Gen3 code path loads the arm as `pybullet_data/kinova_gen3/gen3.urdf`, from
inside the installed `pybullet_data` package. The stock `pybullet` wheel does not
contain that directory, so on a fresh environment each Gen3 simulation, the planner,
and most of the test suite fail with "Cannot load URDF file". This copies the
repository's vendored copy (robot_assets/kinova_gen3/, see its NOTICE.md) to the
place the code looks. Run it once per Python environment:

    python3 scripts/install_robot_assets.py            # install (idempotent)
    python3 scripts/install_robot_assets.py --check    # verify only; exit 1 if wrong
    python3 scripts/install_robot_assets.py --force    # overwrite a differing install

Only the URDF and meshes are copied; LICENSE and NOTICE.md stay in the repository.
An existing install that differs from the vendored files is never overwritten without
--force, because a silently different robot model changes every physics result.
"""
import argparse
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(os.path.dirname(HERE), "robot_assets")
MODELS = ("kinova_gen3",)


def _payload(model_dir):
    """Relative paths of the files that make up the model (urdf + meshes only)."""
    out = []
    for dirpath, _, files in os.walk(model_dir):
        for name in files:
            rel = os.path.relpath(os.path.join(dirpath, name), model_dir)
            if rel in ("LICENSE", "NOTICE.md"):
                continue
            out.append(rel)
    return sorted(out)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _status(src_dir, dst_dir):
    """('missing' | 'identical' | 'different', [problem files])"""
    if not os.path.isdir(dst_dir):
        return "missing", []
    bad = []
    for rel in _payload(src_dir):
        dst = os.path.join(dst_dir, rel)
        if not os.path.isfile(dst) or _sha256(dst) != _sha256(os.path.join(src_dir, rel)):
            bad.append(rel)
    return ("different", bad) if bad else ("identical", [])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify only; exit 1 if not installed")
    ap.add_argument("--force", action="store_true", help="overwrite a differing install")
    args = ap.parse_args()

    try:
        import pybullet_data
    except ImportError:
        sys.exit("pybullet_data not importable -- install requirements.txt first "
                 "(and check `which python3` is the interpreter you installed into).")
    data_dir = pybullet_data.getDataPath()

    failed = False
    for model in MODELS:
        src_dir = os.path.join(SRC_ROOT, model)
        dst_dir = os.path.join(data_dir, model)
        state, bad = _status(src_dir, dst_dir)

        if args.check:
            print(f"{model}: {state}" + (f" ({', '.join(bad)})" if bad else "") + f"  [{dst_dir}]")
            failed |= state != "identical"
            continue
        if state == "identical":
            print(f"{model}: already installed and identical  [{dst_dir}]")
            continue
        if state == "different" and not args.force:
            print(f"{model}: {dst_dir} exists but differs from the vendored copy "
                  f"({', '.join(bad)}). Not overwriting; re-run with --force to replace it.")
            failed = True
            continue
        for rel in _payload(src_dir):
            dst = os.path.join(dst_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(src_dir, rel), dst)
        state, bad = _status(src_dir, dst_dir)
        print(f"{model}: installed ({state})  [{dst_dir}]")
        failed |= state != "identical"

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
