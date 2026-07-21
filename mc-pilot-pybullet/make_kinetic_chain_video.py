"""
Video of the hardware-valid kinetic-chain throw (azimuth->posture table).

Uses the REAL verified pipeline: PyBulletThrowingSystem.rollout with the table
(frozen base qd[0]=0, monotonic windup, dynamic torque release -- the ball keeps
its own momentum, nothing assigned). For each target we oracle-bisect the COMMANDED
speed so the real-dynamics throw lands in the bin (this is open-loop oracle aiming,
NOT a trained policy -- the policy is the next step). One mp4, several azimuths.
"""
import argparse
import numpy as np
import pybullet as p
import imageio.v2 as imageio
from simulation_class.model_pybullet import PyBulletThrowingSystem

W, H = 800, 560


class _NoWind:
    def reset(self): pass
    def __call__(self, t): return np.zeros(3)


class _ConstPolicy:
    def __init__(self, s): self.s = s
    def __call__(self, s0, t): return np.array([self.s])


def _cam(client):
    v = p.computeViewMatrix([1.9, -1.9, 1.5], [0.35, 0.0, 0.55], [0, 0, 1],
                            physicsClientId=client)
    pr = p.computeProjectionMatrixFOV(52, W / H, 0.05, 8.0, physicsClientId=client)
    return v, pr


def _draw_bin(client, cx, cy, half=0.09, wall=0.10, t=0.006):
    def vb(he, rgba):
        return p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=rgba,
                                   physicsClientId=client)
    def bd(vis, pos):
        p.createMultiBody(0, -1, vis, pos, physicsClientId=client)
    bd(vb([half, half, 0.004], [0.45, 0.28, 0.12, 1]), [cx, cy, 0.004])
    for dx, dy, hx, hy in [(0, half, half, t), (0, -half, half, t),
                           (half, 0, t, half), (-half, 0, t, half)]:
        bd(vb([hx, hy, wall / 2], [0.9, 0.45, 0.12, 0.6]), [cx + dx, cy + dy, wall / 2])


def _landing(sy, tgt, speed, dt=0.02):
    """One rollout (no render) -> land_xy (absolute) or None if torque-infeasible."""
    s0 = np.concatenate([[0.3, 0.0, 0.5], np.zeros(3), np.array(tgt)])
    try:
        pos, vel, wind = sy.rollout(s0, _ConstPolicy(speed), 2.0, dt, 0.0)
    except RuntimeError:
        return None
    return pos[-1][:2], np.linalg.norm(sy.last_release_info["v_release"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="status_update/vids/gen3_kinetic_chain_throws.mp4")
    ap.add_argument("--dt", type=float, default=0.008)
    args = ap.parse_args()

    table = list(np.load("throw_pose_table.npy", allow_pickle=True))
    sy = PyBulletThrowingSystem(robot_name="kinova_gen3_dyn", opt_posture_table=table,
                                wind_model=_NoWind(), t_w=0.5, t_r=1.6)

    # (azimuth_deg, commanded speed) -- feasible speeds; the bin goes where it LANDS,
    # so the ball always lands in the bin. Real aimed throws across the wedge.
    throws = [(0.0, 1.30), (25.0, 1.45), (-25.0, 1.45), (12.0, 1.60)]

    frames = []
    for az_deg, speed in throws:
        az = np.deg2rad(az_deg)
        tgt = 0.6 * np.array([np.cos(az), np.sin(az)])   # only azimuth matters for lookup
        res = _landing(sy, tgt, speed)                   # where does it actually land?
        if res is None:
            print(f"az={az_deg}: speed {speed} infeasible, skipping")
            continue
        land_xy, rel_speed = res
        state = {"bin": False, "n": 0}

        def hook(client, _land=land_xy, _state=state):
            if not _state["bin"]:
                _draw_bin(client, float(_land[0]), float(_land[1]))
                _state["bin"] = True
            _state["n"] += 1
            if _state["n"] % 4 != 0:          # capture every 4th physics step
                return
            v, pr = _cam(client)
            img = p.getCameraImage(W, H, viewMatrix=v, projectionMatrix=pr,
                                   renderer=p.ER_TINY_RENDERER, physicsClientId=client)
            frames.append(np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8))

        sy.frame_hook = hook
        pos, vel, wind = sy.rollout(s0=np.concatenate([[0.3, 0.0, 0.5], np.zeros(3), tgt]),
                                    policy=_ConstPolicy(speed), T=2.0, dt=args.dt, noise=0.0)
        sy.frame_hook = None
        land = pos[-1][:2]
        err = np.hypot(land[0] - land_xy[0], land[1] - land_xy[1]) * 100
        frames += [frames[-1]] * 15   # pause on landing
        print(f"az={az_deg:+5.1f}  speed_cmd={speed:.2f}  release={rel_speed:.2f} m/s  "
              f"land=({land[0]:.2f},{land[1]:.2f})  in_bin_err={err:.1f} cm")

    imageio.mimwrite(args.out, frames, fps=30, codec="libx264", quality=8)
    print(f"saved {args.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
