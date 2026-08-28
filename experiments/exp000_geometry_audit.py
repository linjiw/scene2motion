"""EXP-000: the geometry audit every later claim rests on.

Three measurements, each of which would silently invalidate the whole project if wrong.
None of them involve the model; all of them are cheap; all three were surprises.

1. HEAD vs HIGHEST JOINT. ARDY's G1 rig has 34 joints. While standing, its highest is the
   SHOULDER. The real robot's head reaches far higher. If clearance were computed from ARDY
   joint positions -- the obvious thing to do, since that is what the model outputs -- every
   overhead clearance in this project would be optimistic by that gap, which is larger than
   the clearances themselves.

2. COLLISION-CAPSULE COVERAGE. g1.xml ships Unitree's own collision primitives, so it is
   tempting to treat them as ground truth. They are a SIMPLIFICATION of the visual meshes,
   and they under-cover. This measures by how much, per link, by projecting every visual
   mesh vertex against every collision primitive. The number becomes robot.BODY_MARGIN.

3. QPOS REPROJECTION RESIDUAL. ARDY generates free 3D joint rotations; the G1 has single-axis
   hinges. `MujocoQposConverter` projects onto those axes, discarding off-axis rotation. We
   collision-check the REPROJECTED pose (correct: it is the pose the robot can actually hold),
   but the gap to the raw generated motion bounds how much the reference we hand a tracker
   differs from the motion we validated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.robot import ARDY_G1_XML, G1Body  # noqa: E402

# ARDY G1Skeleton34 joint order (ardy/skeleton/definitions.py:286).
ARDY_JOINTS = [
    "pelvis_skel", "left_hip_pitch_skel", "left_hip_roll_skel", "left_hip_yaw_skel",
    "left_knee_skel", "left_ankle_pitch_skel", "left_ankle_roll_skel", "left_toe_base",
    "right_hip_pitch_skel", "right_hip_roll_skel", "right_hip_yaw_skel", "right_knee_skel",
    "right_ankle_pitch_skel", "right_ankle_roll_skel", "right_toe_base",
    "waist_yaw_skel", "waist_roll_skel", "waist_pitch_skel",
    "left_shoulder_pitch_skel", "left_shoulder_roll_skel", "left_shoulder_yaw_skel",
    "left_elbow_skel", "left_wrist_roll_skel", "left_wrist_pitch_skel",
    "left_wrist_yaw_skel", "left_hand_roll_skel",
    "right_shoulder_pitch_skel", "right_shoulder_roll_skel", "right_shoulder_yaw_skel",
    "right_elbow_skel", "right_wrist_roll_skel", "right_wrist_pitch_skel",
    "right_wrist_yaw_skel", "right_hand_roll_skel",
]
# ARDY -> MuJoCo world rotation, det=+1 (ardy/exports/mujoco.py:59, mujoco_to_ardy_matrix).
M_MJ_TO_ARDY = np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])


def surface_dist(body: G1Body, p: np.ndarray, g: int) -> np.ndarray:
    """Distance from points to primitive `g`'s surface (negative inside). Margin excluded."""
    m, d = body.model, body.data
    c, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
    r, t = float(m.geom_size[g][0]), int(m.geom_type[g])
    v = p - c
    if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return np.linalg.norm(v, axis=-1) - r
    a, h = R[:, 2], float(m.geom_size[g][1])
    z = np.clip(v @ a, -h, h)
    return np.linalg.norm(v - z[..., None] * a, axis=-1) - r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="/home/linjiw/ardy/smoke/g1_walk.npz")
    ap.add_argument("--csv", default="/home/linjiw/ardy/smoke/g1_walk.csv")
    ap.add_argument("--out", default="outputs/exp000")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    z = np.load(args.npz, allow_pickle=True)
    qpos = np.loadtxt(args.csv, delimiter=",")
    pj = z["posed_joints"]
    # Measure the raw geometry, with no safety inflation.
    body = G1Body(None, body_margin=0.0)
    m, d = body.model, body.data

    # -- 1. head vs highest joint -------------------------------------------------------
    top_prim = max(body.top_height(q) for q in qpos)
    top_joint = float(pj[:, :, 1].max())
    head = body.head_pos(qpos[0])
    res1 = {"top_of_robot_m": top_prim, "highest_ardy_joint_m": top_joint,
            "gap_m": top_prim - top_joint,
            "head_geom": body.geom_name[body._head_geom],
            "head_center_z_m": float(head[2])}

    # -- 2. capsule coverage ------------------------------------------------------------
    d.qpos[:] = qpos[0]
    mujoco.mj_kinematics(m, d)
    mesh_geoms = [g for g in range(m.ngeom)
                  if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)]
    per_link = []
    for g in mesh_geoms:
        mid = int(m.geom_dataid[g])
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        V = m.mesh_vert[va:va + vn].astype(np.float64)
        P = V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        worst = float(np.min(np.stack([surface_dist(body, P, cg)
                                       for cg in body.robot_geoms]), axis=0).max())
        per_link.append({
            "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]),
            "mesh": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid),
            "protrusion_m": worst})
    per_link.sort(key=lambda r: -r["protrusion_m"])
    res2 = {"n_visual_meshes": len(mesh_geoms), "n_collision_primitives": len(body.robot_geoms),
            "max_protrusion_m": per_link[0]["protrusion_m"],
            "worst_links": per_link[:8]}

    # -- 3. qpos reprojection residual --------------------------------------------------
    pairs = []
    for ji, n in enumerate(ARDY_JOINTS):
        # waist_pitch is the last waist joint before the torso body in the MJCF chain.
        bn = "torso_link" if n == "waist_pitch_skel" else n.replace("_skel", "_link")
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
        if bid >= 0:
            pairs.append((ji, bn, bid))
    per_joint: dict[str, list[float]] = {}
    for t in range(len(qpos)):
        d.qpos[:] = qpos[t]
        mujoco.mj_kinematics(m, d)
        for ji, bn, bid in pairs:
            per_joint.setdefault(bn, []).append(
                float(np.linalg.norm(M_MJ_TO_ARDY @ d.xpos[bid] - pj[t, ji])))
    allv = np.concatenate([np.asarray(v) for v in per_joint.values()])
    res3 = {"n_joints_matched": len(pairs), "n_frames": int(len(qpos)),
            "mean_m": float(allv.mean()), "p95_m": float(np.percentile(allv, 95)),
            "max_m": float(allv.max()),
            "worst_joints": sorted(((k, float(np.max(v))) for k, v in per_joint.items()),
                                   key=lambda kv: -kv[1])[:5]}

    report = {"experiment": "exp000_geometry_audit", "xml": str(ARDY_G1_XML),
              "source_motion": args.npz,
              "head_vs_joint": res1, "capsule_coverage": res2, "qpos_residual": res3,
              "recommended_body_margin_m": round(np.ceil(res2["max_protrusion_m"] * 100) / 100, 3),
              "wall_clock_s": round(time.time() - t0, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"1. top of robot {res1['top_of_robot_m']:.3f} m vs highest ARDY joint "
          f"{res1['highest_ardy_joint_m']:.3f} m -> GAP {res1['gap_m']*100:.1f} cm")
    print(f"2. collision capsules under-cover the meshes by up to "
          f"{res2['max_protrusion_m']*100:.2f} cm "
          f"({res2['n_collision_primitives']} primitives vs {res2['n_visual_meshes']} meshes)")
    for r in res2["worst_links"][:4]:
        print(f"     {r['protrusion_m']*100:+6.2f} cm  {r['body']:26s} {r['mesh']}")
    print(f"3. qpos reprojection residual over {res3['n_joints_matched']} joints: "
          f"mean {res3['mean_m']*1000:.2f} mm, p95 {res3['p95_m']*1000:.2f} mm, "
          f"max {res3['max_m']*1000:.2f} mm")
    print(f"-> BODY_MARGIN = {report['recommended_body_margin_m']} m")


if __name__ == "__main__":
    main()
