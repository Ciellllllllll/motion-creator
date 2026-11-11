"""SOMA 骨格の作成と、ポーズ配列 ⇔ Blender のポーズの変換。

ポーズ配列は [関節数][7]。各関節は親に対するローカル変換
[tx, ty, tz, qx, qy, qz, qw]。関節 0（Hips）はワールド（Z 上）での配置。
"""

import json

import bpy
from mathutils import Matrix, Quaternion, Vector

from . import rig_transfer as rt

SKEL_KEY = "mb_skeleton"
_TAIL_CHILD = {"Hips": "Spine1", "Chest": "Neck1", "Head": "HeadEnd",
               "LeftHand": "LeftHandMiddle1", "RightHand": "RightHandMiddle1"}
_cache = {}


def _mat(t):
    """[tx,ty,tz,qx,qy,qz,qw] → 4x4 行列。"""
    return Matrix.LocRotScale(Vector(t[:3]), Quaternion((t[6], t[3], t[4], t[5])), None)


def _xform(m):
    q = m.to_quaternion()
    p = m.translation
    return [p.x, p.y, p.z, q.x, q.y, q.z, q.w]


def world_matrices(parents, pose):
    """ローカル変換の並びから、各関節のワールド行列を求める。"""
    out = []
    for i, t in enumerate(pose):
        local = _mat(t)
        out.append(local if parents[i] < 0 else out[parents[i]] @ local)
    return out


def skeleton_of(arm):
    return json.loads(arm[SKEL_KEY])


def _info(arm):
    """骨格ごとの固定データ（関節の基本姿勢とボーンの基本行列の差など）。"""
    key = arm.data.name
    hit = _cache.get(key)
    if hit and hit["raw"] == arm[SKEL_KEY]:
        return hit
    skel = skeleton_of(arm)
    names, parents = skel["joint_names"], skel["parents"]
    rest_world = world_matrices(parents, skel["rest"])
    bones = arm.data.bones
    info = {
        "raw": arm[SKEL_KEY], "names": names, "parents": parents, "rest": skel["rest"],
        # C_j：関節の座標系からボーンの座標系へ（基本姿勢で測った固定の差）
        "C": [rest_world[i].inverted() @ bones[n].matrix_local for i, n in enumerate(names)],
        "rest_rel_inv": [((bones[names[parents[i]]].matrix_local.inverted() @ bones[n].matrix_local)
                          if parents[i] >= 0 else bones[n].matrix_local.copy()).inverted()
                         for i, n in enumerate(names)],
    }
    info["C_inv"] = [c.inverted() for c in info["C"]]
    _cache[key] = info
    return info


def build_armature(skel: dict, name: str, color, collection=None):
    """SOMA 骨格のアーマチュアを作る。ボーン名は関節名、ワールドは Z 上。"""
    names, parents = skel["joint_names"], skel["parents"]
    world = world_matrices(parents, skel["rest"])
    heads = [m.translation.copy() for m in world]
    children = {i: [] for i in range(len(names))}
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)

    data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, data)
    (collection or bpy.context.scene.collection).objects.link(arm)
    view_layer = bpy.context.view_layer
    prev_active = view_layer.objects.active
    view_layer.objects.active = arm
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')
    ebones = []
    for i, n in enumerate(names):
        eb = data.edit_bones.new(n)
        eb.head = heads[i]
        tail = None
        kids = children[i]
        want = _TAIL_CHILD.get(n)
        pick = next((k for k in kids if names[k] == want), kids[0] if kids else None)
        if pick is not None and (heads[pick] - heads[i]).length > 1e-4:
            tail = heads[pick]
        if tail is None:
            p = parents[i]
            d = (heads[i] - heads[p]) if p >= 0 else Vector((0, 0, 1))
            d = d.normalized() if d.length > 1e-6 else Vector((0, 0, 1))
            tail = heads[i] + d * (0.02 if "Hand" in n else 0.05)
        eb.tail = tail
        ebones.append(eb)
    for i, p in enumerate(parents):
        if p >= 0:
            ebones[i].parent = ebones[p]
            ebones[i].use_connect = False
    bpy.ops.object.mode_set(mode='OBJECT')
    view_layer.objects.active = prev_active

    arm[SKEL_KEY] = json.dumps(skel)
    arm.show_in_front = True
    data.display_type = 'OCTAHEDRAL'
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    set_color(arm, color)
    return arm


def set_color(arm, color):
    for bone in arm.data.bones:
        try:
            bone.color.palette = 'CUSTOM'
            bone.color.custom.normal = color
            bone.color.custom.select = tuple(min(1.0, c + 0.25) for c in color)
            bone.color.custom.active = tuple(min(1.0, c + 0.4) for c in color)
        except AttributeError:   # 古い Blender
            pass
    arm.color = (*color, 1.0)


def pose_bases(arm, pose, world_inv=None) -> dict:
    """1 ポーズ → {ボーン名: basis 行列}。*pose* の Hips はワールドでの配置。"""
    info = _info(arm)
    mw_inv = world_inv if world_inv is not None else arm.matrix_world.inverted()
    W = world_matrices(info["parents"], pose)
    out, M = {}, []
    for i, n in enumerate(info["names"]):
        m = mw_inv @ W[i] @ info["C"][i]
        M.append(m)
        p = info["parents"][i]
        if p < 0:
            out[n] = info["rest_rel_inv"][i] @ m
        else:
            basis = info["rest_rel_inv"][i] @ M[p].inverted() @ m
            out[n] = basis.to_quaternion().to_matrix().to_4x4()
    return out


def apply_pose(arm, pose, local: bool = False) -> None:
    """*local* が真なら、*pose* はアーマチュアのローカル座標での姿勢。"""
    bases = pose_bases(arm, pose, Matrix.Identity(4) if local else None)
    for pb in arm.pose.bones:
        b = bases.get(pb.name)
        if b is None:
            continue
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = b.to_quaternion()
        pb.location = b.translation if pb.parent is None else (0, 0, 0)
        pb.scale = (1, 1, 1)


def read_pose(arm, local: bool = False) -> list:
    """現在のポーズ（評価済み）をポーズ配列で返す。*local* が真ならアーマチュアのローカル座標。"""
    info = _info(arm)
    mw = Matrix.Identity(4) if local else arm.matrix_world
    pbones = arm.pose.bones
    W = [mw @ pbones[n].matrix @ info["C_inv"][i] for i, n in enumerate(info["names"])]
    out = []
    for i in range(len(W)):
        p = info["parents"][i]
        if p < 0:
            out.append(_xform(W[i]))
        else:
            q = (W[p].inverted() @ W[i]).to_quaternion()
            out.append(list(info["rest"][i][:3]) + [q.x, q.y, q.z, q.w])
    return out


def write_action(arm, frames, name: str, start: int = 1):
    """[N][関節数][7] を新しいアクションにキーフレームとして書き、*arm* に割り当てる。"""
    info = _info(arm)
    mw_inv = arm.matrix_world.inverted()
    bases = [pose_bases(arm, pose, mw_inv) for pose in frames]
    old = arm.animation_data.action if arm.animation_data else None
    if old is not None:
        old.use_fake_user = True
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    action, _, _ = rt._new_action_for(arm, name)
    numbers = list(range(start, start + len(frames)))
    rt.write_bases(arm, action, numbers, bases, info["names"], root_bones=(info["names"][0],))
    action.use_fake_user = True
    return action
