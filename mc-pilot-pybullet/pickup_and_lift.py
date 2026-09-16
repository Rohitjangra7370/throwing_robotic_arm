"""
Move to the recorded pickup pose, close the gripper, verify a real grasp
happened (not closed on nothing), then lift clear of the pickup stand.

Extracted 2026-08-22 from ad hoc use during the first real ball throws --
promoted to a real script because it's now called before every throw, by
both the CLI flow and `throw_gui.py`.

GRASP VERIFICATION, NOT JUST "CLOSE COMMANDED"
------------------------------------------------
A gripper closing on a real ball stalls well short of its fully-closed
target (measured: 58-59%, position AND velocity both dead stable -- see
HardwareThrowExecutor.set_gripper()). Closing on NOTHING reaches ~99-100%
instead. This script checks which happened and REFUSES to lift/proceed if
nothing was grasped, rather than silently continuing to throw an empty hand.

    python3 pickup_and_lift.py --ip 192.168.1.101 --robot kinova_gen3_dyn
"""

import argparse
import json
import sys

import numpy as np
import pybullet as p

from run_hardware_throw import build_arm, make_limits
from robot_arm.kinova_hardware import HardwareThrowExecutor
from robot_arm.robot_profiles import get_robot_profile

# Where a real tennis ball stalls the fingers, measured 2026-08-22: 58.08%
# closed, position AND velocity dead stable. This is the number the grasp
# check has to stay clear of.
BALL_STALL_PCT = 58.08
GRASP_MARGIN_PCT = 10.0      # how far under the COMMANDED close a stall must be
MIN_GRASP_CLOSE = 0.70       # below this the check cannot tell a ball from air


def grasp_threshold_pct(gripper_close, strict=True):
    """
    The "something is between the fingers" threshold for a given commanded
    close amount, in percent closed.

    A gripper closing on NOTHING reaches the position it was commanded to;
    closing on a ball stalls short of it. So the test is "did it stop short of
    what I asked for", and the threshold has to move with the command. It was
    a bare 90.0 for as long as the command was always 100% -- correct then,
    silently wrong the moment the close amount became settable: command 75%,
    close on an empty hand, the fingers stop at 75%, 75 < 90, and the app
    reports a grasp it did not get and throws an empty gripper.

    Below MIN_GRASP_CLOSE the threshold drops into the ball's own stall band
    (58.08%) and the two outcomes stop being distinguishable at all, so a
    close amount that light is refused rather than checked badly. That is a
    limit of the CHECK, not of the gripper -- `strict=False` returns the
    number anyway, for tests and for callers doing their own reasoning.
    """
    close = float(gripper_close)
    if not (0.0 <= close <= 1.0):
        raise ValueError(f"gripper close amount must be in [0.0, 1.0], got {close}")
    thresh = close * 100.0 - GRASP_MARGIN_PCT
    if strict and close < MIN_GRASP_CLOSE:
        raise ValueError(
            f"gripper close amount {close:.2f} puts the grasp check at "
            f"{thresh:.1f}% closed, at or below where a real ball stalls the "
            f"fingers ({BALL_STALL_PCT:.1f}%, measured) -- 'grasped a ball' and "
            f"'closed on nothing' become indistinguishable and the check would "
            f"start passing empty hands. Use >= {MIN_GRASP_CLOSE:.2f}, or "
            f"verify the grasp by eye and bypass this script.")
    return thresh


def pickup_and_lift(ip, robot, pickup_pose_path="pickup_pose.json",
                    lift_z=0.10, home_duration=6.0, lift_duration=4.0,
                    home_speed_frac=0.25, gripper_close=1.0):
    profile = get_robot_profile(robot)
    # Raises before anything moves if the close amount cannot be checked.
    threshold_pct = grasp_threshold_pct(gripper_close)
    q_pickup = np.asarray(json.load(open(pickup_pose_path))["q_pickup_rad"], float)

    arm, _, cid = build_arm(robot)
    limits = make_limits(profile, speed_scale=0.15, arm=arm)
    grasped = False
    pos_pct = None
    try:
        with HardwareThrowExecutor(limits, dry_run=False, ip=ip,
                                   gripper_closed=float(gripper_close)) as ex:
            ex.home(arm, q_pickup, duration=home_duration,
                    speed_frac=home_speed_frac)

            # OPEN FULLY BEFORE CLOSING. Closing a gripper that is already
            # stalled on an object pushes the motor into it a second time, and
            # that is the proximate trigger of the 2026-08-22 ROBOT_IN_FAULT
            # (root cause electrical, but this is what preceded it). It also
            # means the fingers are actually clear when the ball is seated,
            # rather than half-shut around wherever the last cycle left them.
            pre, _ = ex.backend.read_gripper()
            if pre > 1.0:
                print(f"gripper at {pre:.1f}% -- opening fully before the grasp")
                ex.set_gripper(closed=False)
                pre, _ = ex.backend.read_gripper()
                print(f"gripper now {pre:.1f}% (open)")

            ex.set_gripper(closed=True)
            pos_pct, _ = ex.backend.read_gripper()
            grasped = pos_pct < threshold_pct
            print(f"gripper closed at {pos_pct:.1f}% (commanded "
                 f"{gripper_close * 100:.0f}%, grasp under {threshold_pct:.1f}%)  -> "
                 f"{'GRASPED A BALL' if grasped else 'CLOSED ON NOTHING -- no ball present'}")

            if grasped:
                for i, jid in enumerate(profile.joint_ids):
                    p.resetJointState(arm._arm_id, jid, q_pickup[i], physicsClientId=arm._cid)
                ee_pos, ee_orn, _, _ = arm.ee_state()
                target_pos = np.array(ee_pos) + np.array([0, 0, lift_z])
                q_lift = np.array(p.calculateInverseKinematics(
                    arm._arm_id, arm._ee_link, targetPosition=target_pos.tolist(),
                    targetOrientation=ee_orn, restPoses=q_pickup.tolist(),
                    lowerLimits=arm._ik_q_lo.tolist(), upperLimits=arm._ik_q_hi.tolist(),
                    jointRanges=(arm._ik_q_hi - arm._ik_q_lo).tolist(),
                    physicsClientId=arm._cid))[:len(profile.joint_ids)]
                print(f"lifting {lift_z*100:.0f}cm in Z, q_lift: {np.round(q_lift, 4)}")
                ex.home(arm, q_lift, duration=lift_duration,
                        speed_frac=home_speed_frac)
                print("lift done.")
    finally:
        p.disconnect(cid)
    return grasped, pos_pct


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.1.101")
    ap.add_argument("--robot", default="kinova_gen3_dyn")
    ap.add_argument("--pickup_pose", default="pickup_pose.json")
    ap.add_argument("--home_speed_frac", type=float, default=0.25,
                    help="fraction of qd_max for the positioning moves; 0.25 "
                         "default is the bring-up value and is the slowest part "
                         "of a throw cycle")
    ap.add_argument("--lift_z", type=float, default=0.10)
    ap.add_argument("--gripper_close", type=float, default=1.0,
                    help="how far to close the fingers, 0.0 (open) to 1.0 "
                         "(fully closed). 1.0 is the historical behaviour. "
                         "Lower it for a lighter grip on a softer or larger "
                         "ball; below %.2f the grasp check cannot distinguish "
                         "a ball from an empty hand and refuses."
                         % MIN_GRASP_CLOSE)
    args = ap.parse_args()

    grasped, pos_pct = pickup_and_lift(args.ip, args.robot, args.pickup_pose, args.lift_z,
                                  home_speed_frac=args.home_speed_frac,
                                  gripper_close=args.gripper_close)
    sys.exit(0 if grasped else 2)   # distinct exit code: caller (e.g. the GUI) can refuse to throw


if __name__ == "__main__":
    main()
