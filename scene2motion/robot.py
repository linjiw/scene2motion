# Scene2Motion-G1: the robot body model and exact scene collision, via MuJoCo.
#
# Why MuJoCo rather than capsules fitted to ARDY joints
# -----------------------------------------------------
# ARDY's G1 rig has 34 joints whose highest point while standing is the SHOULDER, at
# ~1.07 m. The real G1's head reaches 1.30 m (measured: the torso_link head capsule,
# r=0.068, sits at z=1.232 in the rest pose). A collision model built from ARDY joint
# positions therefore under-estimates the robot's standing height by 23 cm, which is
# larger than every overhead clearance we care about — it would silently declare
# head-first beam collisions "clear".
#
# ARDY already exports MuJoCo qpos (ardy.exports.mujoco.MujocoQposConverter), and the
# shipped g1.xml carries the robot's OWN collision primitives (capsules/spheres/cylinders,
# radii straight from Unitree). So we do forward kinematics in MuJoCo on the exported qpos
# and let MuJoCo do exact primitive-vs-box distance queries. No hand-rolled geometry, no
# convention risk.
#
# Clearance vs penetration comes from one mechanism: scene geoms are given a large contact
# `margin`, so MuJoCo reports near-contacts with a SIGNED distance. dist > 0 is clearance,
# dist < 0 is penetration depth.

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .scenes import Scene

ARDY_G1_XML = Path("/home/linjiw/ardy/ardy/assets/skeletons/g1skel34/xml/g1.xml")

# Contact bitmasks: robot geoms only ever see scene geoms, never each other.
ROBOT_CONTYPE, ROBOT_CONAFF = 1, 2
SCENE_CONTYPE, SCENE_CONAFF = 2, 1

# How far away a scene geom still reports a (positive-distance) near-contact.
# 0.6 m comfortably covers every clearance we report, at negligible cost.
CLEARANCE_MARGIN = 0.6

# g1.xml's collision primitives UNDER-COVER the robot's actual visual meshes: measured over
# all 38 mesh geoms at the rest pose, the worst vertex protrudes 3.56 cm beyond every
# collision primitive, and the head mesh alone protrudes 3.29 cm past the head capsule.
# Since overhead clearance is the channel this project leans on hardest, a sign test on the
# raw primitives would certify head-first collisions as clear by up to 3.3 cm. Every body
# extent and every scene box is therefore inflated by this margin, so "collision free" means
# "clear of the real geometry", not "clear of the simplified capsules".
#   measured by experiments/exp000_geometry_audit.py
BODY_MARGIN = 0.04

# MuJoCo geom types that are part of G1's collision model. Meshes in g1.xml are visual.
# Compared as ints: model.geom_type holds numpy ints, which do not compare equal to the
# pybind11 mjtGeom enum members.
_COLLIDABLE = {int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE),
               int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_BOX)}
_CAPSULE, _CYLINDER, _SPHERE = (int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                                int(mujoco.mjtGeom.mjGEOM_SPHERE))


def _fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def build_scene_xml(scene: Scene | None, xml_path: Path = ARDY_G1_XML,
                    body_margin: float = BODY_MARGIN) -> str:
    """G1 MJCF with `scene`'s boxes welded into the worldbody and contacts switched on.

    The shipped g1.xml sets contype=0/conaffinity=0 on the whole robot (it is an export
    rig, not a sim model), so nothing would ever collide. We re-enable contacts only
    between the robot's collision primitives and the scene, which keeps self-collision out
    of the metric — self-collision is the tracker's problem, not the planner's.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # meshdir is relative to the xml; make it absolute so the model can be built from a string.
    comp = root.find("compiler")
    if comp is not None and comp.get("meshdir"):
        comp.set("meshdir", str((xml_path.parent / comp.get("meshdir")).resolve()))

    for geom in root.iter("geom"):
        gtype = geom.get("type")
        if gtype in ("capsule", "sphere", "cylinder", "box"):
            geom.set("contype", str(ROBOT_CONTYPE))
            geom.set("conaffinity", str(ROBOT_CONAFF))
        elif gtype == "plane":
            # the ground plane: keep it inert, the floor is not an obstacle
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    wb = root.find("worldbody")
    if scene is not None:
        for i, b in enumerate(scene.boxes):
            ET.SubElement(wb, "geom", {
                "name": f"scene_{i}_{b.label or 'box'}",
                "type": "box",
                "pos": _fmt(b.center),
                "size": _fmt(np.asarray(b.half) + body_margin),
                "contype": str(SCENE_CONTYPE),
                "conaffinity": str(SCENE_CONAFF),
                "margin": str(CLEARANCE_MARGIN),
                "gap": "0",
                "rgba": "0.6 0.6 0.65 0.4",
            })
    return ET.tostring(root, encoding="unicode")


@dataclass
class FrameContact:
    frame: int
    robot_geom: str
    scene_geom: str
    dist: float  # signed: >0 clearance, <0 penetration depth
    # Contact normal's vertical component, oriented from the ROBOT geom toward the scene
    # geom. Near +1 the obstacle is directly overhead; near 0 it is beside the robot.
    normal_z: float = 0.0

    @property
    def overhead(self) -> bool:
        """True when lowering the body would increase this clearance.

        The duck channel can only buy headroom. A squeeze between a wall and a beam edge
        produces the same small `dist` as a head-vs-beam near-miss, but crouching does
        nothing for it -- so a repair loop driven by the undifferentiated minimum would
        command deep crouches against lateral deficits it cannot fix. Within 60 degrees of
        straight up is counted as overhead.
        """
        return self.normal_z >= 0.5


class G1Body:
    """Exact G1 collision queries against one scene, driven by ARDY-exported qpos."""

    def __init__(self, scene: Scene | None = None, xml_path: Path = ARDY_G1_XML,
                 body_margin: float = BODY_MARGIN):
        self.body_margin = body_margin
        self.model = mujoco.MjModel.from_xml_string(
            build_scene_xml(scene, xml_path, body_margin))
        self.data = mujoco.MjData(self.model)
        self.scene = scene
        m = self.model
        self.robot_geoms = [g for g in range(m.ngeom)
                            if int(m.geom_type[g]) in _COLLIDABLE
                            and m.geom_contype[g] == ROBOT_CONTYPE]
        self.scene_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] == SCENE_CONTYPE]
        # Many collision primitives are unnamed in g1.xml; fall back to the owning body so
        # reports say "left_elbow_link" rather than "geom50".
        self.geom_name = {}
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if not nm:
                body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "?"
                nm = f"{body}#{g}"
            self.geom_name[g] = nm
        # g1.xml declares explicit <pair> contacts for every foot pad against the ground
        # plane. Explicit pairs BYPASS contype/conaffinity filtering, so the floor keeps
        # colliding no matter what masks we set. We do not fight it: foot-vs-floor
        # penetration is a useful physical-consistency signal, so we keep those contacts
        # and simply report them separately from scene collisions.
        self.floor_geoms = {g for g in range(m.ngeom)
                            if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)}
        self._head_geom = self._find_head_geom()

    def _find_head_geom(self) -> int:
        """The topmost collision primitive on torso_link — G1's head capsule."""
        m = self.model
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        cands = [g for g in self.robot_geoms if m.geom_bodyid[g] == tid]
        return max(cands, key=lambda g: m.geom_pos[g][2])

    # -- kinematics -------------------------------------------------------------------

    def fk(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)

    def body_points(self, qpos: np.ndarray) -> np.ndarray:
        """World positions of every robot collision primitive at this pose, (G,3)."""
        self.fk(qpos)
        return self.data.geom_xpos[self.robot_geoms].copy()

    def _extent(self, g: int, n: np.ndarray) -> tuple[float, float]:
        """Exact (min, max) projection of geom `g`'s SURFACE onto unit direction `n`.

        MuJoCo stores a capsule/cylinder as size=(radius, half_length) about its local +z,
        so the surface extent is centre +/- half_length*|a.n| +/- radius, not centre +/-
        radius. Getting this wrong under-reports the head by its half-length and the limbs
        by most of their length.
        """
        m, d = self.model, self.data
        c = float(d.geom_xpos[g] @ n)
        r = float(m.geom_size[g][0]) + self.body_margin
        t = int(m.geom_type[g])
        if t == _SPHERE:
            return c - r, c + r
        a = d.geom_xmat[g].reshape(3, 3)[:, 2]      # the primitive's own long axis
        an = float(a @ n)
        h = float(m.geom_size[g][1])
        if t == _CAPSULE:
            reach = abs(an) * h + r                  # spherical caps: full radius always
        else:  # cylinder: flat caps, so the radial term shrinks as the axis aligns with n
            reach = abs(an) * h + r * float(np.sqrt(max(0.0, 1.0 - an * an)))
        return c - reach, c + reach

    _UP = np.array([0.0, 0.0, 1.0])

    def top_height(self, qpos: np.ndarray) -> float:
        """True top-of-robot height: highest primitive SURFACE, not highest joint."""
        self.fk(qpos)
        return float(max(self._extent(g, self._UP)[1] for g in self.robot_geoms))

    def head_pos(self, qpos: np.ndarray) -> np.ndarray:
        self.fk(qpos)
        return self.data.geom_xpos[self._head_geom].copy()

    def half_width(self, qpos: np.ndarray, normal: np.ndarray | None = None) -> float:
        """Half-extent of the body about the pelvis along a horizontal direction.

        `normal` defaults to world +y. For a gap the relevant direction is the one
        PERPENDICULAR TO TRAVEL, which is not the world y-axis once the robot turns or
        sidles — measuring along a fixed axis would credit a sidling robot with a
        narrowness it does not have.
        """
        n = self._UP * 0 + (np.array([0.0, 1.0, 0.0]) if normal is None else np.asarray(normal, float))
        n = n / (np.linalg.norm(n) + 1e-12)
        self.fk(qpos)
        pel = float(self.data.qpos[:3] @ n)
        return float(max(max(abs(lo - pel), abs(hi - pel))
                         for lo, hi in (self._extent(g, n) for g in self.robot_geoms)))

    def swept_half_width(self, qpos_traj: np.ndarray, normal: np.ndarray,
                         frames: slice | None = None) -> float:
        """Worst half-width over a window, measured along a fixed horizontal `normal`.

        This is the quantity a gap actually tests: the widest the body ever gets while
        passing through, not its width at one instant.
        """
        sl = frames or slice(None)
        return float(max(self.half_width(q, normal) for q in qpos_traj[sl]))

    # -- collision --------------------------------------------------------------------

    def frame_contacts(self, qpos: np.ndarray, frame: int = 0
                       ) -> tuple[list[FrameContact], list[FrameContact]]:
        """(scene_contacts, floor_contacts) at one pose, with signed distances.

        dist > 0 is clearance (reported out to CLEARANCE_MARGIN); dist < 0 is penetration
        depth. Duplicate contact points from box-vs-capsule manifolds are kept — the
        summary only ever takes their minimum.
        """
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        m = self.model
        scene_c: list[FrameContact] = []
        floor_c: list[FrameContact] = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if m.geom_contype[g1] == ROBOT_CONTYPE:
                rg, og = g1, g2
            elif m.geom_contype[g2] == ROBOT_CONTYPE:
                rg, og = g2, g1
            else:
                continue
            # c.frame[:3] is the contact normal pointing from geom1 to geom2; orient it to
            # point away from the robot so a positive z means "obstacle above".
            nz = float(c.frame[2]) * (1.0 if rg == g1 else -1.0)
            fc = FrameContact(frame, self.geom_name[rg], self.geom_name[og], float(c.dist), nz)
            (floor_c if og in self.floor_geoms else scene_c).append(fc)
        return scene_c, floor_c

    def trajectory_report(self, qpos_traj: np.ndarray) -> dict:
        """Whole-body collision summary for a (T, 36) ARDY-exported qpos trajectory.

        Scene collision and foot-vs-floor penetration are reported separately: the first is
        the planning metric (did the body hit the world), the second is a physical-
        consistency signal about the generated motion itself (feet sinking into the ground
        is the classic kinematic-motion artefact and predicts tracker failure).
        """
        T = len(qpos_traj)
        min_clear = np.full(T, np.nan)
        over_clear = np.full(T, np.nan)
        pen = np.zeros(T)
        floor_pen = np.zeros(T)
        culprit: list[str] = [""] * T
        worst = (0.0, -1, "", "")
        for t in range(T):
            scene_c, floor_c = self.frame_contacts(qpos_traj[t], t)
            if floor_c:
                floor_pen[t] = max(0.0, -min(c.dist for c in floor_c))
            if not scene_c:
                continue
            worst_c = min(scene_c, key=lambda c: c.dist)
            min_clear[t] = worst_c.dist
            over = [c for c in scene_c if c.overhead]
            if over:
                over_clear[t] = min(c.dist for c in over)
            culprit[t] = worst_c.robot_geom
            if worst_c.dist < 0:
                pen[t] = -worst_c.dist
                if pen[t] > worst[0]:
                    worst = (pen[t], t, worst_c.robot_geom, worst_c.scene_geom)
        seen = min_clear[~np.isnan(min_clear)]
        pen_frames = int((pen > 0).sum())
        return {
            "collision_free": bool(pen_frames == 0),
            "penetration_frames": pen_frames,
            "penetration_fraction": float(pen_frames / max(T, 1)),
            "max_penetration_m": float(pen.max()),
            "mean_penetration_m": float(pen[pen > 0].mean()) if pen_frames else 0.0,
            # Frames where nothing was within CLEARANCE_MARGIN are genuinely clear; treat
            # them as margin-distance rather than dropping them from the minimum.
            "min_clearance_m": float(seen.min()) if len(seen) else float(CLEARANCE_MARGIN),
            "worst": {"depth_m": worst[0], "frame": worst[1],
                      "robot_geom": worst[2], "scene_geom": worst[3]},
            "per_frame_min_clearance": np.where(np.isnan(min_clear), CLEARANCE_MARGIN,
                                                min_clear).tolist(),
            # The headroom component alone: what a duck schedule is able to act on.
            "per_frame_overhead_clearance": np.where(np.isnan(over_clear), CLEARANCE_MARGIN,
                                                     over_clear).tolist(),
            "culprit_geoms": sorted({c for c in culprit if c}),
            "max_foot_floor_penetration_m": float(floor_pen.max()),
            "mean_foot_floor_penetration_m": float(floor_pen.mean()),
        }
