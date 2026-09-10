"""
Annotated, high-quality video of the TRAINED opt_pose (kinetic-chain) Gen3
policy -- same real rollout as make_trained_kinetic_chain_video.py (real
trained weights, real opt_posture_table throw, no bespoke mechanics), but
with end-effector / ball / target labels overlaid and a higher-resolution,
anti-aliased render.

Labels are drawn in image space by projecting known world-frame points
through the camera's view/projection matrices -- pybullet's DIRECT-mode
TinyRenderer does not rasterize addUserDebugText/Line into getCameraImage,
so this cannot be done with pybullet debug draw calls.
"""
import argparse
import os
import pickle as pkl

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data
import torch
from PIL import Image, ImageDraw, ImageFont

import policy_learning.Policy as Policy
from robot_arm.robot_profiles import get_robot_profile
from simulation_class.model_pybullet import PyBulletThrowingSystem

# Body creation order inside PyBulletThrowingSystem.rollout is fixed:
# plane (loadURDF) -> arm (ArmController.__init__ -> single loadURDF) -> ball
# (createMultiBody). draw_bin() below only adds bodies *after* the first
# frame, so these indices are stable for the lifetime of one rollout.
PLANE_BODY, ARM_BODY, BALL_BODY = 0, 1, 2

SSAA = 2  # supersample factor for anti-aliasing (TinyRenderer has no AA)
OUT_W, OUT_H = 1600, 896  # divisible by 16, avoids an ffmpeg macro-block resize
RENDER_W, RENDER_H = OUT_W * SSAA, OUT_H * SSAA

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_policy(log_path, cfg):
    with open(os.path.join(log_path, "log.pkl"), "rb") as f:
        log = pkl.load(f)
    st = log["parameters_trial_list"][-1]
    pol = Policy.Throwing_Policy(
        full_state_dim=8, target_dim=2, num_basis=st["centers"].shape[0], u_max=cfg["uM"],
        lengthscales_init=st["log_lengthscales"].exp().numpy()[0],
        centers_init=st["centers"].numpy(), weight_init=st["f_linear.weight"].numpy(),
        flg_drop=False, dtype=torch.float64, device=torch.device("cpu"))
    pol.load_state_dict(st)
    pol.eval()
    return pol, len(log["parameters_trial_list"])


# eye, target, up, fov_deg -- defined for a floor-mounted arm (base_height=0);
# cam() shifts the z components by the checkpoint's actual base_height so
# framing stays centered on the arm whether it sits on the floor or a plate.
# "iso" matches the original unlabeled video's camera.
CAMERA_PRESETS = {
    "iso":   ([2.1, -2.1, 1.5], [0.9, 0.0, 0.15], [0, 0, 1], 52),
    "front": ([3.0, 0.0, 0.8], [0.6, 0.0, 0.3], [0, 0, 1], 46),
    "side":  ([0.6, -2.6, 0.8], [0.6, 0.0, 0.3], [0, 0, 1], 46),
    "top":   ([0.6, 0.0, 3.0], [0.6, 0.0, 0.0], [1, 0, 0], 52),
}

_CAM_VIEW = "iso"      # set from --view in main()
_BASE_HEIGHT = 0.0     # set from cfg["base_height"] in main()


def cam(cid):
    eye, target, up, fov = CAMERA_PRESETS[_CAM_VIEW]
    eye = [eye[0], eye[1], eye[2] + _BASE_HEIGHT]
    target = [target[0], target[1], target[2] + _BASE_HEIGHT]
    view = p.computeViewMatrix(eye, target, up, physicsClientId=cid)
    proj = p.computeProjectionMatrixFOV(fov, RENDER_W / RENDER_H, 0.05, 7.0, physicsClientId=cid)
    return view, proj


def project(view, proj, world_xyz):
    """World-frame point -> (pixel_x, pixel_y) in the RENDER_W x RENDER_H frame."""
    v = np.array(view, dtype=np.float64).reshape(4, 4, order="F")
    pr = np.array(proj, dtype=np.float64).reshape(4, 4, order="F")
    clip = pr @ v @ np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0])
    if abs(clip[3]) < 1e-9:
        return None
    ndc = clip[:3] / clip[3]
    px = (ndc[0] * 0.5 + 0.5) * RENDER_W
    py = (1.0 - (ndc[1] * 0.5 + 0.5)) * RENDER_H
    return px, py


def draw_bin(cid, cx, cy, half=0.09, wall=0.10, t=0.006):
    def vb(he, rgba):
        return p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=rgba, physicsClientId=cid)

    def bd(vis, pos):
        p.createMultiBody(0, -1, vis, pos, physicsClientId=cid)

    bd(vb([half, half, 0.004], [0.45, 0.28, 0.12, 1]), [cx, cy, 0.004])
    for dx, dy, hx, hy in [(0, half, half, t), (0, -half, half, t),
                           (half, 0, t, half), (-half, 0, t, half)]:
        bd(vb([hx, hy, wall / 2], [0.9, 0.45, 0.12, 0.6]), [cx + dx, cy + dy, wall / 2])


def annotate(rgb_array, labels):
    """labels: list of (px, py, text, dot_rgb). Anchors that land close together
    (ball at the gripper before release, ball at the target on landing) get
    their text boxes pushed apart so they never overlap into an unreadable
    merged string."""
    img = Image.fromarray(rgb_array)
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.truetype(_FONT_PATH, 26 * SSAA)
    pad = 4 * SSAA
    line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 2 * pad
    placed_boxes = []

    def overlaps(bbox):
        return any(bbox[0] < b[2] and bbox[2] > b[0] and bbox[1] < b[3] and bbox[3] > b[1]
                  for b in placed_boxes)

    for entry in labels:
        if entry is None:
            continue
        px, py, text, dot_rgb = entry
        if not (0 <= px <= RENDER_W and 0 <= py <= RENDER_H):
            continue
        r = 7 * SSAA
        draw.ellipse([px - r, py - r, px + r, py + r], fill=dot_rgb + (255,),
                     outline=(0, 0, 0, 255), width=max(1, SSAA))
        tx, ty = px + 14 * SSAA, py - 14 * SSAA
        bbox = draw.textbbox((tx, ty), text, font=font)
        bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        while overlaps(bbox):
            ty += line_h
            bbox = draw.textbbox((tx, ty), text, font=font)
            bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        placed_boxes.append(bbox)
        draw.rectangle(bbox, fill=(0, 0, 0, 160))
        draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))
        draw.line([px, py, tx, ty], fill=dot_rgb + (200,), width=max(1, SSAA))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--log_path", default="results_kinetic_chain_gen3_tcp/1",
                    help="current preferred checkpoint: TCP-offset-corrected training "
                         "(see CLAUDE.md -- results_kinetic_chain_gen3/* is historical, "
                         "understates release speed ~26%%)")
    ap.add_argument("--opt_pose", default="throw_pose_table_tcp.npy")
    ap.add_argument("--n_throws", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--view", choices=sorted(CAMERA_PRESETS), default="iso",
                    help="camera preset: iso (default, matches the original video), "
                         "front (facing the throw direction), side (profile), "
                         "top (overhead)")
    ap.add_argument("--out", default=None,
                    help="default: status_update/vids/gen3_kinetic_chain_labeled_<view>.mp4")
    ap.add_argument("--tool_offset_z", type=float, default=0.12,
                    help="TCP offset the --opt_pose table was searched at; must "
                         "match its stamp (see run_hardware_throw.py).")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"status_update/vids/gen3_kinetic_chain_labeled_{args.view}.mp4"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    global _CAM_VIEW, _BASE_HEIGHT
    _CAM_VIEW = args.view

    with open(os.path.join(args.log_path, "config_log.pkl"), "rb") as f:
        cfg = pkl.load(f)
    _BASE_HEIGHT = float(cfg.get("base_height", 0.0))
    pol, n_trials = load_policy(args.log_path, cfg)
    print(f"Loaded trained policy: {n_trials} trials logged, robot={cfg['robot_name']}, "
         f"base_height={_BASE_HEIGHT}")

    profile = get_robot_profile(cfg["robot_name"])
    RELEASE_POS = np.array(cfg["release_pos"], dtype=float)

    table = list(np.load(args.opt_pose, allow_pickle=True))
    launch_deg = float(table[0]["elev_deg"])
    from run_hardware_throw import _check_tool_offset_matches_table
    _check_tool_offset_matches_table(table, args.tool_offset_z)

    def policy(s, t):
        with torch.no_grad():
            inp = torch.tensor(np.asarray(s), dtype=torch.float64).unsqueeze(0)
            return np.array([float(pol(inp, t=0, p_dropout=0.0).item())])

    _az0 = min(table, key=lambda e: abs(e["azimuth_deg"]))
    _cid0 = p.connect(p.DIRECT)
    _arm0 = p.loadURDF(
        pybullet_data.getDataPath() + "/" + profile.urdf_rel_path,
        useFixedBase=True, physicsClientId=_cid0)
    for j in range(7):
        p.resetJointState(_arm0, j, _az0["q"][j], physicsClientId=_cid0)
    release_xy = np.array(
        p.getLinkState(_arm0, profile.ee_link, computeForwardKinematics=True,
                       physicsClientId=_cid0)[4][:2])
    p.disconnect(_cid0)

    lm, lM, gM = cfg["lm"], cfg["lM"], cfg["gM"]
    f_lo, f_hi = lm - release_xy[0], lM - release_xy[0]
    rng = np.random.default_rng(args.seed)

    def sample_target():
        flight = rng.uniform(f_lo, f_hi)
        beta = rng.uniform(-gM, gM)
        return release_xy + flight * np.array([np.cos(beta), np.sin(beta)])

    targets = [sample_target() for _ in range(args.n_throws)]

    frames = []
    errs = []
    for i, tgt in enumerate(targets):
        sysm = PyBulletThrowingSystem(
            mass=cfg["ball_mass"], radius=cfg["ball_radius"],
            launch_angle_deg=launch_deg, arm_noise=None,
            t_w=cfg["T_W"], t_r=cfg["T_R"], robot_name=cfg["robot_name"],
            target_height=cfg["target_height"], opt_posture_table=table,
            opt_launch_deg=launch_deg,
            base_height=float(cfg.get("base_height", 0.0)),
            tool_offset=[0.0, 0.0, args.tool_offset_z])

        fr, drawn = [], {"d": False}

        def cap(cid, _fr=fr, _d=drawn, _t=tgt):
            if not _d["d"]:
                draw_bin(cid, _t[0], _t[1])
                _d["d"] = True
            if len(_fr) % 2:
                _fr.append(None)
                return
            view, proj = cam(cid)
            img = p.getCameraImage(RENDER_W, RENDER_H, viewMatrix=view, projectionMatrix=proj,
                                   renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
            rgb = np.reshape(img[2], (RENDER_H, RENDER_W, 4))[:, :, :3].astype(np.uint8)

            ee_pos = p.getLinkState(ARM_BODY, profile.ee_link, computeForwardKinematics=True,
                                    physicsClientId=cid)[4]
            ball_pos, _ = p.getBasePositionAndOrientation(BALL_BODY, physicsClientId=cid)

            def make_label(world_xyz, text, dot_rgb):
                px_py = project(view, proj, world_xyz)
                return None if px_py is None else (px_py[0], px_py[1], text, dot_rgb)

            labels = [
                make_label(ee_pos, "end-effector", (60, 140, 255)),
                make_label(ball_pos, "ball", (255, 220, 40)),
                make_label((_t[0], _t[1], 0.004), "target", (255, 90, 90)),
            ]
            rgb = annotate(rgb, labels)
            small = Image.fromarray(rgb).resize((OUT_W, OUT_H), Image.LANCZOS)
            _fr.append(np.asarray(small))

        sysm.frame_hook = cap
        s0 = np.concatenate([RELEASE_POS, np.zeros(3), tgt])
        pos, vel, wind = sysm.rollout(s0, policy, T=cfg["T"], dt=cfg["Ts"], noise=0.0)
        land = pos[-1][:2]
        err = float(np.linalg.norm(land - tgt))
        errs.append(err)
        speed = float(np.linalg.norm(sysm.last_release_info["v_release"])) \
            if sysm.last_release_info is not None else float("nan")
        print(f"throw {i+1}/{args.n_throws}: target=({tgt[0]:.3f},{tgt[1]:.3f}) "
             f"landed=({land[0]:.3f},{land[1]:.3f}) err={err*100:.1f}cm "
             f"release_speed={speed:.3f}m/s")

        real = [f for f in fr if f is not None]
        frames += real + [real[-1]] * 25

    imageio.mimwrite(args.out, frames, fps=50, codec="libx264", quality=None,
                     output_params=["-crf", "16", "-preset", "slow"])
    errs = np.array(errs)
    print(f"\nmean err {errs.mean()*100:.1f}cm  median {np.median(errs)*100:.1f}cm  "
         f"max {errs.max()*100:.1f}cm  hit<10cm {100*np.mean(errs<0.10):.0f}%")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
