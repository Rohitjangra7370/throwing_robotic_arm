"""
One command for the whole run-day start: start_of_day.py's GO/NO-GO gate,
then (only on GO) launch hardware_session.py with this rig's standard
checkpoint config. Two existing tools, called in sequence -- no logic
duplicated from either.

    python3 run_day.py                 # gate, then launch the session GUI
    python3 run_day.py --gate-only      # just the gate, don't launch the GUI
    python3 run_day.py --ip 192.168.1.101 --dry-run   # empty-gripper rehearsal

Always invoked with /usr/bin/python3 explicitly below, regardless of which
interpreter ran this launcher -- bare `python3` on this machine can resolve to
a Conda base env with no torch/pybullet/cv2 (see CLAUDE.md's gotcha).
"""
import argparse
import subprocess
import sys

PYTHON = "/usr/bin/python3"

# Current preferred checkpoint (HARDWARE_RUNBOOK.md / CLAUDE.md "Which checkpoint"):
# trained under TCP-offset-corrected physics, fresh-seed eval 1.90cm mean.
CHECKPOINT_ARGS = [
    "--robot", "kinova_gen3_dyn",
    "--log_path", "results_kinetic_chain_gen3_tcp/1",
    "--opt_pose", "throw_pose_table_tcp.npy",
    "--tool_offset_z", "0.12",
    "--base_height", "0.433",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="192.168.1.101")
    ap.add_argument("--gate-only", action="store_true",
                    help="run start_of_day.py's checks and stop -- don't launch the session GUI")
    ap.add_argument("--dry-run", action="store_true",
                    help="launch hardware_session.py WITHOUT --arm (no real motion)")
    args = ap.parse_args()

    print("=== run_day: start_of_day.py gate ===")
    gate = subprocess.run([PYTHON, "start_of_day.py", "--ip", args.ip])
    if gate.returncode != 0:
        print("\nrun_day: NO-GO -- stopping here. Fix what start_of_day.py flagged and re-run.")
        return 1

    if args.gate_only:
        print("\nrun_day: GO -- --gate-only set, not launching the session GUI.")
        return 0

    print("\nrun_day: GO -- launching hardware_session.py")
    cmd = [PYTHON, "hardware_session.py", "--ip", args.ip] + CHECKPOINT_ARGS
    if not args.dry_run:
        cmd.append("--arm")
    else:
        print("run_day: --dry-run set -- no real arm motion in the throw cycle "
              "(pickup_and_lift still moves the real arm/gripper, unconditionally).")
    subprocess.run(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
