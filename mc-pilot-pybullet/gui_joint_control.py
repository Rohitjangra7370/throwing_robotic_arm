"""Real PyBullet GUI with a slider per joint, driving the actual Kinova Gen3
URDF used throughout this project (pybullet_data/kinova_gen3/gen3.urdf).
No hand-reconstructed FK -- this is the real simulator's own rendering.

Run: python gui_joint_control.py
Sliders appear in the PyBullet window itself (left panel). Console prints the
current joint angles (deg + rad) once a second so you can read them off.
"""
import time
import numpy as np
import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

arm = p.loadURDF(pybullet_data.getDataPath() + "/kinova_gen3/gen3.urdf", useFixedBase=True)
n_joints = p.getNumJoints(arm)

q_neutral = [-0.01, 0.382, -0.04, 1.46, -0.012, 0.731, 0.0]

sliders = []
names = []
for j in range(7):
    info = p.getJointInfo(arm, j)
    name = info[1].decode()
    lo, hi = info[8], info[9]
    if lo >= hi:            # continuous joints report lo>hi; give a practical range
        lo, hi = -np.pi, np.pi
    s = p.addUserDebugParameter(name, lo, hi, q_neutral[j])
    sliders.append(s)
    names.append(name)

p.resetBasePositionAndOrientation(arm, [0, 0, 0], [0, 0, 0, 1])
for j in range(7):
    p.resetJointState(arm, j, q_neutral[j])

# ball at the end effector so you can see release-point/reach context
ee_link = 7
ee_pos = p.getLinkState(arm, ee_link, computeForwardKinematics=True)[4]
ball = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=-1,
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.033, rgbaColor=[1, 1, 0, 1]),
    basePosition=ee_pos,
)

print("PyBullet GUI running. Drag sliders in the window to pose the arm.", flush=True)
print("Joint order:", names, flush=True)
last_print = 0.0
try:
    while True:
        q = [p.readUserDebugParameter(s) for s in sliders]
        for j in range(7):
            p.resetJointState(arm, j, q[j])
        ee_pos = p.getLinkState(arm, ee_link, computeForwardKinematics=True)[4]
        p.resetBasePositionAndOrientation(ball, ee_pos, [0, 0, 0, 1])

        now = time.time()
        if now - last_print > 1.0:
            last_print = now
            deg = np.degrees(q)
            print(f"\ndeg = {np.round(deg, 1).tolist()}", flush=True)
            print(f"rad = np.array({np.round(q, 4).tolist()})", flush=True)
            print(f"EE pos = {np.round(ee_pos, 3).tolist()}", flush=True)

        p.stepSimulation()
        time.sleep(1.0 / 120.0)
except p.error:
    print("Window closed, exiting cleanly.", flush=True)
