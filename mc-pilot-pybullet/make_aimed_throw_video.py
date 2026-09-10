"""
Video of the CURRENT working throw: the AIMED real-dynamics throw.
Torque control + grip constraint + physical release + MEASURED velocity (no assignment),
using the direction-constrained aimed pose (find_throw_pose.py). Runs at dt=0.005 (the
stable regime for this throw), throws at a few azimuths, ball lands in a bin at the target.
"""
import numpy as np
import pybullet as p
import pybullet_data
import imageio.v2 as imageio
from scipy.optimize import linprog
from robot_arm.arm_controller import ArmController
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model import _ball_accel
import os

_VIDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "status_update", "vids")

QD = np.array([1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218])
QN = np.array([-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0])
MASS, RAD, W, H = 0.0577, 0.0327, 1024, 720


def cam(cid):
    v = p.computeViewMatrix([1.8, -1.8, 1.15], [0.45, 0.0, 0.35], [0, 0, 1], physicsClientId=cid)
    return v, p.computeProjectionMatrixFOV(50, W / H, 0.05, 6.0, physicsClientId=cid)


def basket(cid, c, half=0.08, wall=0.09, t=0.006):
    cx, cy = float(c[0]), float(c[1])
    def vb(he, rgba): return p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=rgba, physicsClientId=cid)
    def bd(v, pos): p.createMultiBody(0, -1, v, pos, physicsClientId=cid)
    bd(vb([half, half, .004], [.45, .28, .12, 1]), [cx, cy, .004])
    for dx, dy, hx, hy in [(0, half, half, t), (0, -half, half, t), (half, 0, t, half), (-half, 0, t, half)]:
        bd(vb([hx, hy, wall/2], [.9, .45, .12, .55]), [cx+dx, cy+dy, wall/2])


def aimed_qd(J, d):
    d = d/np.linalg.norm(d); c = np.zeros(8); c[-1] = -1
    r = linprog(c, A_eq=np.hstack([J, -d.reshape(3, 1)]), b_eq=np.zeros(3),
                bounds=[(-QD[i], QD[i]) for i in range(7)]+[(0, None)], method="highs")
    return (r.x[-1], r.x[:7]) if r.success else (0.0, np.zeros(7))


def throw(pose, az_deg, frames, dt=0.005):
    cid = p.connect(p.DIRECT); p.setGravity(0, 0, -9.81, physicsClientId=cid); p.setTimeStep(dt, physicsClientId=cid)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid); p.loadURDF("plane.urdf", physicsClientId=cid)
    prof = get_robot_profile("kinova_gen3_dyn")
    arm = ArmController(cid, pybullet_data.getDataPath()+"/"+prof.urdf_rel_path, robot_name="kinova_gen3_dyn"); arm.reset()
    qrel = np.array(pose["q"]).copy(); qrel[0] = pose["q"][0] + np.deg2rad(az_deg)
    qrel = qrel + 2*np.pi*np.round((QN - qrel)/(2*np.pi))
    qf = arm._ik_q_neutral.copy()
    for li, dof in enumerate(arm._dof_ids): qf[dof] = qrel[li]
    jl, _ = p.calculateJacobian(arm._arm_id, arm._ee_link, [0, 0, 0], qf.tolist(), [0.]*arm._n_dofs, [0.]*arm._n_dofs, physicsClientId=cid)
    J = np.array(jl)[:, arm._dof_ids]
    a = np.deg2rad(pose["elev_deg"]); azr = np.deg2rad(az_deg)
    d = np.array([np.cos(a)*np.cos(azr), np.cos(a)*np.sin(azr), np.sin(a)])
    smax, qd = aimed_qd(J, d)
    relpos = np.array(p.getLinkState(arm._arm_id, arm._ee_link, computeForwardKinematics=True, physicsClientId=cid)[4])
    for j in range(arm._n_dofs): p.resetJointState(arm._arm_id, j, arm._ik_q_neutral[j], physicsClientId=cid)
    ball = p.createMultiBody(MASS, p.createCollisionShape(p.GEOM_SPHERE, radius=RAD, physicsClientId=cid),
                             p.createVisualShape(p.GEOM_SPHERE, radius=RAD, rgbaColor=[.9, .85, .1, 1], physicsClientId=cid),
                             arm.ee_state()[0].tolist(), physicsClientId=cid)
    p.changeDynamics(ball, -1, linearDamping=0, angularDamping=0, physicsClientId=cid); arm.attach_ball(ball)
    coeffs, _, _, _ = arm.plan_throw(d, relpos, t_w=0.5, t_r=1.6, T=3.0, q_release_override=qrel, qd_release_override=qd)
    rel_step = int(coeffs["t_r"]/dt); drawn = False; vr = None
    def grab():
        vw, pr = cam(cid); img = p.getCameraImage(W, H, viewMatrix=vw, projectionMatrix=pr, renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
        frames.append(np.reshape(img[2], (H, W, 4))[:, :, :3].astype(np.uint8))
    for step in range(int((coeffs["t_r"]+1.4)/dt)):
        t = step*dt
        if step < rel_step:
            qt, qdt, qddt = arm.get_setpoint(coeffs, t, with_accel=True); arm.step(qt, qdt, qddt)
        elif step == rel_step:
            vr = arm.release_ball(ball, dynamic=True, keep_collision_disabled=True)
            relp = np.array(p.getBasePositionAndOrientation(ball, physicsClientId=cid)[0])
            x = relp.copy(); v = vr.copy()
            for _ in range(3000):
                v = v + _ball_accel(x, v, MASS, RAD, np.zeros(3))*0.002; x = x+v*0.002
                if x[2] <= 0 and v[2] < 0: break
            basket(cid, x[:2]); drawn = True
        else:
            bp = np.array(p.getBasePositionAndOrientation(ball, physicsClientId=cid)[0]); bv = np.array(p.getBaseVelocity(ball, physicsClientId=cid)[0])
            f = MASS*(_ball_accel(bp, bv, MASS, RAD, np.zeros(3)) - np.array([0, 0, -9.81])); p.applyExternalForce(ball, -1, f.tolist(), [0, 0, 0], p.WORLD_FRAME, physicsClientId=cid)
            if bp[2] < RAD+0.006 and bv[2] < 0: break
        p.stepSimulation(physicsClientId=cid)
        if step % 4 == 0: grab()
    for _ in range(12): grab()
    print(f"az={az_deg:+d}: measured release speed {np.linalg.norm(vr):.2f} m/s")
    p.disconnect(cid)


if __name__ == "__main__":
    pose = np.load("throw_pose.npy", allow_pickle=True).item()
    frames = []
    for az in [0, 18, -18]:
        throw(pose, az, frames)
    out = os.path.join(_VIDS, "gen3_aimed_throw_current.mp4")
    imageio.mimwrite(out, frames, fps=40, codec="libx264", quality=8)
    print("saved", out)
