"""
REAL-DYNAMICS throw test: torque-control the Gen3 to the optimized release posture
with velocity-limit-optimal joint scheduling, carry the ball on a grip CONSTRAINT,
release it dynamically, and MEASURE the ball's actual velocity + landing.

Nothing is assigned: the ball keeps whatever momentum the physics gives it at release
(arm.release_ball(dynamic=True) removes the grip constraint, no set_vel). This is the
honest test of whether the ~1 m throw survives real dynamics.
"""
import argparse, os
import numpy as np
import pybullet as p
import pybullet_data
import imageio.v2 as imageio
from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model import _ball_accel

_VIDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "status_update", "vids")

QD = np.array([1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218])
MASS, RAD = 0.0577, 0.0327
W, H = 1024, 720


def cam(cid):
    v = p.computeViewMatrix([2.0, -2.0, 1.35], [0.7, 0.0, 0.2], [0, 0, 1], physicsClientId=cid)
    return v, p.computeProjectionMatrixFOV(52, W / H, 0.05, 7.0, physicsClientId=cid)


def basket(cid, c, half=0.09, wall=0.09, t=0.006):
    cx, cy = float(c[0]), float(c[1])
    def vb(he, rgba): return p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=rgba, physicsClientId=cid)
    def bd(v, pos): p.createMultiBody(0, -1, v, pos, physicsClientId=cid)
    bd(vb([half, half, .004], [.45, .28, .12, 1]), [cx, cy, .004])
    for dx, dy, hx, hy in [(0, half, half, t), (0, -half, half, t), (half, 0, t, half), (-half, 0, t, half)]:
        bd(vb([hx, hy, wall/2], [.9, .45, .12, .55]), [cx+dx, cy+dy, wall/2])


def run(q_release, elev_deg, out, make_vid=True, dt=0.005):
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setTimeStep(dt, physicsClientId=cid)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)
    prof = get_robot_profile("kinova_gen3_dyn")
    urdf = pybullet_data.getDataPath() + "/" + prof.urdf_rel_path
    arm = ArmController(cid, urdf, robot_name="kinova_gen3_dyn")
    arm.reset()

    # velocity-optimal qd at the optimized release posture
    q_full = arm._ik_q_neutral.copy()
    for li, dof in enumerate(arm._dof_ids):
        q_full[dof] = q_release[li]
    jl, _ = p.calculateJacobian(arm._arm_id, arm._ee_link, [0, 0, 0], q_full.tolist(),
                                [0.]*arm._n_dofs, [0.]*arm._n_dofs, physicsClientId=cid)
    J = np.array(jl)[:, arm._dof_ids]
    a = np.deg2rad(elev_deg); d = np.array([np.cos(a), 0, np.sin(a)])
    qd_opt = QD * np.sign(d @ J)
    ee_rel_pos = J @ np.zeros(7)  # placeholder; get real FK below

    # ball, grip constraint
    ee_pos_init, _, _, _ = arm.ee_state()
    bcol = p.createCollisionShape(p.GEOM_SPHERE, radius=RAD, physicsClientId=cid)
    bvis = p.createVisualShape(p.GEOM_SPHERE, radius=RAD, rgbaColor=[.9, .85, .1, 1], physicsClientId=cid)
    ball = p.createMultiBody(MASS, bcol, bvis, ee_pos_init.tolist(), physicsClientId=cid)
    p.changeDynamics(ball, -1, linearDamping=0, angularDamping=0, physicsClientId=cid)
    arm.attach_ball(ball)

    # plan throw to the optimized posture with velocity-optimal qd (real overrides)
    t_w, t_r, t_arm = 0.5, 1.6, 3.0
    coeffs, q_rel, qd_rel, v_planned = arm.plan_throw(
        d, ee_pos_init, t_w=t_w, t_r=t_r, T=t_arm,
        q_release_override=q_release, qd_release_override=qd_opt)
    t_r_actual = coeffs["t_r"]
    release_step = int(t_r_actual / dt)
    total = int((t_r_actual + 2.0) / dt)

    frames = []
    basket_drawn = {"d": False}
    released = False
    v_release = None; rel_pos = None; land = None
    for step in range(total):
        t = step * dt
        if not released:
            q_t, qd_t, qdd_t = arm.get_setpoint(coeffs, t, with_accel=True)
            arm.step(q_t, qd_t, qdd_t)
            if step >= release_step:
                ee_pos, _, _, _ = arm.ee_state()
                rel_pos = ee_pos.copy()
                v_release = arm.release_ball(ball, dynamic=True, keep_collision_disabled=True)
                released = True
                # predict landing to place basket
                x = rel_pos.copy(); v = v_release.copy()
                for _ in range(3000):
                    v = v + _ball_accel(x, v, MASS, RAD, np.zeros(3)) * 0.002; x = x + v * 0.002
                    if x[2] <= 0 and v[2] < 0: break
                land = x.copy()
                if make_vid: basket(cid, land[:2]); basket_drawn["d"] = True
        else:
            bpos = np.array(p.getBasePositionAndOrientation(ball, physicsClientId=cid)[0])
            bvel = np.array(p.getBaseVelocity(ball, physicsClientId=cid)[0])
            f = MASS * (_ball_accel(bpos, bvel, MASS, RAD, np.zeros(3)) - np.array([0, 0, -9.81]))
            p.applyExternalForce(ball, -1, f.tolist(), [0, 0, 0], p.WORLD_FRAME, physicsClientId=cid)
            if bpos[2] < RAD + 0.005 and bvel[2] < 0:
                land = bpos.copy()
        p.stepSimulation(physicsClientId=cid)
        if make_vid and step % 3 == 0:
            v_, pr = cam(cid)
            img = p.getCameraImage(W, H, viewMatrix=v_, projectionMatrix=pr, renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
            frames.append(np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8))

    speed = np.linalg.norm(v_release)
    throw = np.hypot(land[0] - rel_pos[0], land[1] - rel_pos[1])
    v_plan_speed = np.linalg.norm(J @ qd_rel)
    print(f"planned (kinematic J*qd): {v_plan_speed:.2f} m/s | "
          f"MEASURED ball release speed: {speed:.2f} m/s | THROW distance: {throw*100:.1f} cm | "
          f"clip_scale={coeffs['clip_scale']:.2f} time_scale={coeffs['time_scale']:.2f}")
    if make_vid and frames:
        frames += [frames[-1]] * 25
        imageio.mimwrite(out, frames, fps=40, codec="libx264", quality=8)
        print(f"saved {out}")
    p.disconnect(cid)
    return speed, throw, v_plan_speed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--elev", type=float, default=35.0)
    ap.add_argument("--out", default=os.path.join(_VIDS, "gen3_REAL_dynamics_throw.mp4"))
    args = ap.parse_args()
    posts = np.load("/tmp/claude-1000/-home-olympusforge-trade-throwing-robotic-arm/743253bb-cb0a-4a1f-be7d-7d13d7807b2a/scratchpad/best_postures.npy")
    run(posts[args.idx], args.elev, args.out)
