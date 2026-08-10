"""
Repair URDF links that declare no inertial block.

THE DEFECT
----------
`pybullet_data/kinova_gen3/gen3.urdf` declares its three camera frames as empty
self-closing tags:

    <link name="camera_link" />
    <link name="camera_depth_frame" />
    <link name="camera_color_frame" />

A URDF link with no `<inertial>` block is massless by the spec, but PyBullet
substitutes **mass = 1 kg, inertia = diag(1,1,1)** and only warns on stdout:

    No inertial data for link, using mass=1, localinertiadiagonal = 1,1,1

All three hang off the wrist, so the model carries **3 kg of phantom mass at the
end of the arm** -- on a 6.5 kg arm. It cannot be undone after loading: the
links are attached by FIXED joints, so PyBullet merges their inertia into the
parent at load time and a later `changeDynamics` updates the reported mass
without touching the merged inertia the dynamics solver actually uses.

MEASURED IMPACT (lab Gen3, 2026-08-07)
--------------------------------------
Static gravity torque compared against the arm's own torque sensors at a
measured pose:

    with phantoms   : 2.1-2.4x the arm's reading
    phantoms zeroed : within 17-27%

And on the shipped throw's peak torque:

    with phantoms   : 36.7 Nm = 94.2% of joint 1's limit  -> precheck REFUSES
    phantoms zeroed : 16.1 Nm = 41.4%                     -> precheck passes

WHY THIS IS SAFE TO CHANGE
--------------------------
The planned MOTION is bit-identical either way -- T, t_r, |v|, q_release and
qd_release all agree to 0.000e+00 -- because `plan_throw` stretches time on
VELOCITY feasibility, not torque. Torque is a validation gate, never a driver.
So this corrects what the model *reports* without altering what the arm *does*,
and every accuracy number in the ledger is untouched. What it does change is
every torque figure, which is the point: they were wrong.

The repair is deliberately narrow -- it only supplies an explicit ZERO-mass
inertial block to links that declared none, which is what the URDF spec says
they already meant. It invents no masses and touches nothing else.
"""

from __future__ import annotations

import hashlib
import os
import re

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_urdf_cache")

# An explicit zero-mass inertial block: what a link with no <inertial> means per
# the URDF spec, stated in the form PyBullet will not override.
_ZERO_INERTIAL = (
    '<inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.0"/>'
    '<inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/></inertial>'
)

_EMPTY_LINK = re.compile(r'<link\s+name="([^"]+)"\s*/>')


def massless_links(urdf_text: str):
    """Names of links declared with no body (hence no inertial block)."""
    return _EMPTY_LINK.findall(urdf_text)


def repair_massless_links(urdf_path: str, cache_dir: str | None = None,
                          verbose: bool = False) -> str:
    """
    Path to a URDF whose empty links carry an explicit zero-mass inertial block.

    Returns `urdf_path` unchanged when there is nothing to repair, so callers can
    use it unconditionally. Mesh filenames are rewritten to absolute paths, since
    the repaired copy does not sit beside the original's `meshes/` directory.
    """
    try:
        with open(urdf_path, "r") as f:
            text = f.read()
    except OSError:
        return urdf_path

    names = massless_links(text)
    if not names:
        return urdf_path

    src_dir = os.path.dirname(os.path.abspath(urdf_path))
    fixed = text
    for nm in names:
        fixed = fixed.replace(f'<link name="{nm}" />',
                              f'<link name="{nm}">{_ZERO_INERTIAL}</link>')
        fixed = fixed.replace(f'<link name="{nm}"/>',
                              f'<link name="{nm}">{_ZERO_INERTIAL}</link>')

    def _abs(m):
        fn = m.group(1)
        if os.path.isabs(fn) or "://" in fn:
            return m.group(0)
        return 'filename="%s"' % os.path.join(src_dir, fn)

    fixed = re.sub(r'filename="([^"]+)"', _abs, fixed)

    cache_dir = cache_dir or _CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    tag = hashlib.sha1((os.path.abspath(urdf_path) + fixed).encode()).hexdigest()[:12]
    out = os.path.join(cache_dir, f"{os.path.basename(urdf_path)[:-5]}.{tag}.urdf")
    if not os.path.exists(out):
        with open(out, "w") as f:
            f.write(fixed)
        if verbose:
            print(f"[urdf] repaired {len(names)} massless link(s) {names} -> {out}")
    return out
