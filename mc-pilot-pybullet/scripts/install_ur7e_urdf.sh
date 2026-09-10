#!/usr/bin/env bash
#
# Build pybullet_data/ur7e/ from the official Universal Robots ROS 2 description.
#
# THIS SCRIPT IS THE PROVENANCE RECORD for the UR7e model. The generated URDF
# and meshes live inside the installed `pybullet_data` package (outside this
# repo, and therefore not version-controlled), exactly like `kinova_gen3/` --
# so this script, not the output, is what makes the arm reproducible.
#
# Idempotent: re-running overwrites the vendored copy with an identical one.
#
#   usage:  bash scripts/install_ur7e_urdf.sh
#
# ---------------------------------------------------------------------------
# WHERE THE MODEL COMES FROM, AND WHAT IS ACTUALLY UR7e ABOUT IT
# ---------------------------------------------------------------------------
# Source: github.com/UniversalRobots/Universal_Robots_ROS2_Description, tag
# 4.3.1, `config/ur7e/`. Pin the TAG: the repo's default `ros2` branch has NO
# ur7e at all (its config/ stops at ur5e), which is what most search results
# and older checkouts point at.
#
# Diffed against config/ur5e/ at that tag:
#   * physical_parameters.yaml  -- BYTE-IDENTICAL to ur5e's
#   * default_kinematics.yaml   -- UR5e's numbers (0.1625 / -0.425 / -0.3922 /
#                                  0.1333 / 0.0997 / 0.0996)
#   * visual_parameters.yaml    -- points at meshes/ur5e/; there is no
#                                  meshes/ur7e/ in the repo
#   * joint_limits.yaml         -- header cites the *UR5e* user manual
#
# That is mostly legitimate rather than a stub. UR themselves publish ONE
# shared "Robot UR5e/UR7e" JT file and ONE shared working-area PDF; both arms
# are 850 mm reach, 20.6 kg, D151 mm footprint. Same mechanics, hotter joints.
# The 180 deg/s velocity limit independently matches the UR7e tech sheet.
#
# What is NOT verified for the UR7e, and must not be reported as measured:
#   * max_effort 150/150/150/28/28/28 Nm  -- UR5e values. UR's public
#     max-joint-torque article has no UR7e row. The UR7e lifts 7.5 kg vs 5, so
#     the real limits are likely higher (i.e. our precheck fails closed, which
#     is the safe direction) -- but measure via RTDE actual_current /
#     target_moment before publishing a torque-headroom number for this arm.
#   * link masses and inertia tensors -- ur5e's, verbatim.
#
# ---------------------------------------------------------------------------
# PHANTOM MASS
# ---------------------------------------------------------------------------
# The generated URDF declares SIX bodyless links (world, base_link, ft_frame,
# base, flange, tool0). PyBullet silently substitutes mass=1 kg,
# inertia=diag(1,1,1) for each, and three of them (ft_frame, flange, tool0)
# hang off the wrist. Loaded raw this arm weighs 26.700 kg against a real
# 21.700 kg of declared links -- 5 phantom kg, 3 of them at the end effector.
# `robot_arm/urdf_fixup.py::repair_massless_links` fixes it (same defect the
# Gen3's three camera frames had, where it was worth 2.1-2.4x on gravity
# torque and made a valid throw fail its precheck). Every loader in this repo
# already routes through that repair; do not load this URDF directly.
#
set -euo pipefail

TAG=4.3.1
REPO=UniversalRobots/Universal_Robots_ROS2_Description
ROS_SETUP=${ROS_SETUP:-/opt/ros/humble/setup.bash}

command -v curl >/dev/null || { echo "need curl" >&2; exit 1; }
[ -f "$ROS_SETUP" ] || {
    echo "need a ROS 2 install for xacro (looked for $ROS_SETUP; override with ROS_SETUP=)" >&2
    exit 1
}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "[1/5] fetching $REPO @ $TAG"
curl -fsSL "https://github.com/$REPO/archive/refs/tags/$TAG.tar.gz" \
    | tar xz -C "$WORK"
SRC="$WORK/Universal_Robots_ROS2_Description-$TAG"
[ -d "$SRC/config/ur7e" ] || { echo "tag $TAG has no config/ur7e" >&2; exit 1; }

# ur.urdf.xacro resolves its own defaults through $(find ur_description), which
# needs an ament index entry. Build a throwaway overlay rather than installing
# the package -- keeps this repo's environment untouched.
echo "[2/5] staging ament overlay for xacro"
mkdir -p "$WORK/overlay/share/ament_index/resource_index/packages"
touch "$WORK/overlay/share/ament_index/resource_index/packages/ur_description"
ln -sfn "$SRC" "$WORK/overlay/share/ur_description"

echo "[3/5] expanding xacro -> URDF"
bash -lc "source '$ROS_SETUP' \
    && export AMENT_PREFIX_PATH='$WORK/overlay:\$AMENT_PREFIX_PATH' \
    && xacro '$SRC/urdf/ur.urdf.xacro' name:=ur7e ur_type:=ur7e" \
    > "$WORK/ur7e_raw.urdf"

# The expansion carries no <ros2_control> block (that lives in the separate
# _driver package), so the result is plain URDF and needs no stripping.
if grep -q "ros2_control" "$WORK/ur7e_raw.urdf"; then
    echo "unexpected ros2_control block in generated URDF -- strip it before use" >&2
    exit 1
fi

DEST=$(/usr/bin/python3 -c "import pybullet_data; print(pybullet_data.getDataPath())")/ur7e
echo "[4/5] vendoring into $DEST"
mkdir -p "$DEST/meshes/visual" "$DEST/meshes/collision"
# Meshes are ur5e's by upstream's own reference -- see the note above.
cp "$SRC"/meshes/ur5e/visual/*.dae     "$DEST/meshes/visual/"
cp "$SRC"/meshes/ur5e/collision/*.stl  "$DEST/meshes/collision/"
# Rewrite package:// refs to paths relative to the URDF, matching how
# kinova_gen3/gen3.urdf refers to its own meshes/.
sed 's#package://ur_description/meshes/ur5e/#meshes/#g' \
    "$WORK/ur7e_raw.urdf" > "$DEST/ur7e.urdf"

echo "[5/5] verifying load + phantom-mass repair"
cd "$(dirname "$0")/.."
/usr/bin/python3 - <<'PY'
import os, sys
import pybullet as p, pybullet_data
sys.path.append(".")
from robot_arm.urdf_fixup import repair_massless_links, massless_links

raw = os.path.join(pybullet_data.getDataPath(), "ur7e/ur7e.urdf")
empties = massless_links(open(raw).read())
fixed = repair_massless_links(raw)
assert fixed != raw, "expected massless links to need repair"

cid = p.connect(p.DIRECT)
rid = p.loadURDF(fixed, useFixedBase=True, physicsClientId=cid)
n = p.getNumJoints(rid, physicsClientId=cid)
mass = sum(p.getDynamicsInfo(rid, j, physicsClientId=cid)[0] for j in range(n))
rev = [j for j in range(n)
       if p.getJointInfo(rid, j, physicsClientId=cid)[2] == p.JOINT_REVOLUTE]
tool0 = [j for j in range(n)
         if p.getJointInfo(rid, j, physicsClientId=cid)[12] == b"tool0"][0]

assert rev == [2, 3, 4, 5, 6, 7], f"actuated joint ids moved: {rev}"
assert tool0 == 10, f"tool0 link index moved: {tool0}"
assert abs(mass - 21.700) < 1e-3, f"mass {mass:.3f} != 21.700 (phantom mass?)"
print(f"      repaired {len(empties)} bodyless link(s): {empties}")
print(f"      joint_ids={rev}  ee_link(tool0)={tool0}  mass={mass:.3f} kg  OK")
PY

echo
echo "done. profiles 'ur7e' / 'ur7e_dyn' in robot_arm/robot_profiles.py now load."
