import json
import math
import pathlib
import random

from mathutils import Quaternion, Vector

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def skeleton():
    return json.loads((DATA / "skeleton.json").read_text(encoding="utf-8"))


def fk_cases():
    return json.loads((DATA / "fk_expected.json").read_text(encoding="utf-8"))["cases"]


def walk_frames():
    return json.loads((DATA / "walk_frames.json").read_text(encoding="utf-8"))["frames"]


def random_pose(skel, seed=0, max_deg=60.0):
    rnd = random.Random(seed)
    pose = [list(j) for j in skel["rest"]]
    for j in pose:
        axis = (rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1))
        dq = Quaternion(axis, math.radians(rnd.uniform(-max_deg, max_deg)))
        q = Quaternion((j[6], j[3], j[4], j[5])) @ dq
        j[3:7] = [q.x, q.y, q.z, q.w]
    pose[0][0:3] = [0.7, -1.3, 0.9]
    return pose


def quat_angle_deg(a, b):
    qa = Quaternion((a[6], a[3], a[4], a[5]))
    qb = Quaternion((b[6], b[3], b[4], b[5]))
    if qa.dot(qb) < 0:   # q と -q は同じ回転
        qb.negate()
    d = qa.normalized().rotation_difference(qb.normalized())
    return math.degrees(2.0 * math.asin(min(1.0, Vector((d.x, d.y, d.z)).length)))


def world_heads(arm):
    import bpy
    bpy.context.view_layer.update()
    return {pb.name: arm.matrix_world @ pb.head for pb in arm.pose.bones}


def character_path():
    path = DATA / "player_mixamo_repaired.fbx"
    assert path.is_file(), f"テスト用キャラクターを置いてください: {path}"
    return path


def load_character(path=None):
    """Mixamo キャラクターの FBX を読み込み、アーマチュアを返す。"""
    import bpy
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path or character_path()))
    new = [o for o in bpy.data.objects if o not in before]
    return next(o for o in new if o.type == "ARMATURE")


def soma_with_walk(name="MB_src"):
    from motionbricks_blender import soma_rig
    arm = soma_rig.build_armature(skeleton(), name, (1, 1, 1))
    action = soma_rig.write_action(arm, walk_frames(), name + "_walk")
    return arm, action


def action_signature(action):
    from motionbricks_blender import rig_transfer as rt
    out = []
    for fc in rt.action_fcurves(action) or []:
        out.append((fc.data_path, fc.array_index, tuple(round(k.co[1], 6) for k in fc.keyframe_points)))
    return sorted(out)
