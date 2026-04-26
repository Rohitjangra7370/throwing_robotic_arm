"""
Inspect supported robot-arm profiles without importing PyBullet.

This script extracts robot-dependent variables directly from the URDF/XML files
and writes paper-friendly CSV + Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import xml.etree.ElementTree as ET

from robot_arm.robot_profiles import iter_profiles


def _pybullet_data_root() -> str:
    here = os.path.expanduser(
        r"~\AppData\Local\Programs\Python\Python310\lib\site-packages\pybullet_data"
    )
    if os.path.isdir(here):
        return here
    raise FileNotFoundError(
        "Could not locate pybullet_data. Update inspect_robot_arms.py with the correct path."
    )


def _joint_rows(urdf_path: str):
    root = ET.parse(urdf_path).getroot()
    rows = []
    link_order = []
    for elem in root:
        if elem.tag == "link":
            link_order.append(elem.attrib["name"])
        if elem.tag != "joint":
            continue
        joint_type = elem.attrib["type"]
        joint_name = elem.attrib["name"]
        limit = elem.find("limit")
        parent = elem.find("parent")
        child = elem.find("child")
        rows.append(
            {
                "joint_name": joint_name,
                "joint_type": joint_type,
                "child_link": child.attrib["link"] if child is not None else "",
                "parent_link": parent.attrib["link"] if parent is not None else "",
                "lower": float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else math.nan,
                "upper": float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else math.nan,
                "velocity": float(limit.attrib["velocity"]) if limit is not None and "velocity" in limit.attrib else math.nan,
            }
        )
    return rows, link_order


def main():
    parser = argparse.ArgumentParser("Inspect robot-dependent arm variables")
    parser.add_argument("--out_dir", type=str, default="results_robot_arm_profiles")
    args = parser.parse_args()

    pybullet_root = _pybullet_data_root()
    os.makedirs(args.out_dir, exist_ok=True)

    summary_rows = []
    detail_rows = []
    for profile in iter_profiles():
        urdf_path = os.path.join(pybullet_root, *profile.urdf_rel_path.split("/"))
        joints, link_order = _joint_rows(urdf_path)
        revolute = [j for j in joints if j["joint_type"] == "revolute"]
        actuated = [revolute[i] for i in range(min(len(profile.joint_ids), len(revolute)))]
        qd_max = [j["velocity"] for j in actuated]
        lower = [j["lower"] for j in actuated]
        upper = [j["upper"] for j in actuated]
        span = [u - l for l, u in zip(lower, upper)]

        summary_rows.append(
            {
                "robot": profile.name,
                "urdf": profile.urdf_rel_path,
                "actuated_dofs": len(profile.joint_ids),
                "ee_link": profile.ee_link,
                "default_release_x": profile.default_release_pos[0],
                "default_release_z": profile.default_release_pos[2],
                "speed_min": profile.speed_bounds[0],
                "speed_max": profile.speed_bounds[1],
                "mean_profile_vel_limit_rad_s": sum(profile.qd_max) / len(profile.qd_max),
                "max_profile_vel_limit_rad_s": max(profile.qd_max),
                "mean_urdf_vel_limit_rad_s": sum(qd_max) / len(qd_max),
                "mean_joint_range_rad": sum(span) / len(span),
                "notes": profile.notes,
            }
        )

        for joint_id, row in zip(profile.joint_ids, actuated):
            detail_rows.append(
                {
                    "robot": profile.name,
                    "joint_id": joint_id,
                    "joint_name": row["joint_name"],
                    "joint_type": row["joint_type"],
                    "lower": row["lower"],
                    "upper": row["upper"],
                    "velocity_limit": row["velocity"],
                    "child_link": row["child_link"],
                }
            )

    summary_csv = os.path.join(args.out_dir, "robot_profile_summary.csv")
    detail_csv = os.path.join(args.out_dir, "robot_profile_joint_details.csv")
    md_path = os.path.join(args.out_dir, "robot_profile_summary.md")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Robot | DOFs | EE Link | Release (x,z) m | Speed Bounds m/s | Mean Profile Vel Limit rad/s | Mean URDF Vel Limit rad/s | Mean Joint Range rad |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['robot']} | {row['actuated_dofs']} | {row['ee_link']} | "
                f"({row['default_release_x']:.2f}, {row['default_release_z']:.2f}) | "
                f"{row['speed_min']:.2f}-{row['speed_max']:.2f} | "
                f"{row['mean_profile_vel_limit_rad_s']:.3f} | {row['mean_urdf_vel_limit_rad_s']:.3f} | "
                f"{row['mean_joint_range_rad']:.3f} |\n"
            )

    print(f"Wrote {summary_csv}")
    print(f"Wrote {detail_csv}")
    print(f"Wrote {md_path}")
    print("\nSummary:")
    for row in summary_rows:
        print(
            f"  {row['robot']}: dofs={row['actuated_dofs']}, ee_link={row['ee_link']}, "
            f"speed={row['speed_min']:.2f}-{row['speed_max']:.2f} m/s, "
            f"mean profile vel limit={row['mean_profile_vel_limit_rad_s']:.3f} rad/s"
        )


if __name__ == "__main__":
    main()
