"""手足の編集。

「手足を編集」で、選んだポーズ人形に手首・足首のハンドルと、ひじ・ひざの向きを
決める目印（どちらも操作用のボーン）を足してポーズモードに入る。ハンドルを動かすと
2 本の骨（上腕＋前腕、太もも＋すね）を解き直す。Blender の IK 制約は使わない
（付けた瞬間に姿勢が変わるため）。動かしていない間は姿勢が 1 mm も変わらない。
"""

import json
import math

import bpy
from bpy.app.handlers import persistent
from mathutils import Matrix, Quaternion, Vector

from . import poses, soma_rig

# (ハンドル, 目印, 付け根, 中間, 先端, 目印の既定の向き：+1 前 / -1 後ろ)
LIMBS = (
    ("MB_左手", "MB_左ひじ", "LeftArm", "LeftForeArm", "LeftHand", -1.0),
    ("MB_右手", "MB_右ひじ", "RightArm", "RightForeArm", "RightHand", -1.0),
    ("MB_左足", "MB_左ひざ", "LeftLeg", "LeftShin", "LeftFoot", 1.0),
    ("MB_右足", "MB_右ひざ", "RightLeg", "RightShin", "RightFoot", 1.0),
)
HANDLE_NAMES = tuple(n for limb in LIMBS for n in limb[:2])
HANDLE_COLOR = (1.0, 1.0, 1.0)
POLE_COLOR = (1.0, 0.85, 0.2)
POLE_DISTANCE = 0.35
REACH = 0.999
_solving = False


def _forward(pb):
    """腰の左右の関節から求めた体の正面（アーマチュア座標）。"""
    right = pb["RightLeg"].head - pb["LeftLeg"].head
    fwd = Vector((0, 0, 1)).cross(right)
    return fwd.normalized() if fwd.length > 1e-6 else Vector((0, -1, 0))


def _bend_direction(H, K, A, fallback):
    axis = A - H
    if axis.length < 1e-6:
        return fallback
    axis.normalize()
    perp = (K - H) - axis * (K - H).dot(axis)
    if perp.length < 0.01:          # ほぼまっすぐ：既定の向き
        perp = fallback - axis * fallback.dot(axis)
    return perp.normalized() if perp.length > 1e-6 else fallback


def _set_color(bone, color):
    try:
        bone.color.palette = 'CUSTOM'
        bone.color.custom.normal = color
        bone.color.custom.select = color
        bone.color.custom.active = color
    except AttributeError:
        pass


HANDLE_SHAPE, POLE_SHAPE = "MB_ハンドル形", "MB_目印形"
HANDLE_SIZE, POLE_SIZE = 0.065, 0.04


def _shape(name):
    """ハンドルの見た目（線だけのメッシュ）。シーンには出さず、ボーンの形としてだけ使う。"""
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type == 'MESH':
        return obj
    verts, edges = [], []
    if name == HANDLE_SHAPE:        # 3 方向の輪でできた球
        n = 20
        for axis in range(3):
            start = len(verts)
            for k in range(n):
                a = 2 * math.pi * k / n
                p = [0.0, 0.0, 0.0]
                p[(axis + 1) % 3], p[(axis + 2) % 3] = math.cos(a), math.sin(a)
                verts.append(tuple(p))
                edges.append((start + k, start + (k + 1) % n))
    else:                           # ひし形
        verts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        edges = [(a, b) for a in range(6) for b in range(a + 1, 6) if a // 2 != b // 2]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, [])
    return bpy.data.objects.new(name, mesh)


def _add_handles(arm):
    pb = arm.pose.bones
    fwd = _forward(pb)
    spots = {}
    for handle, pole, root, mid, tip, sign in LIMBS:
        H, K, A = pb[root].head.copy(), pb[mid].head.copy(), pb[tip].head.copy()
        spots[handle] = A
        spots[pole] = K + _bend_direction(H, K, A, fwd * sign) * POLE_DISTANCE
    bpy.ops.object.mode_set(mode='EDIT')
    for name, pos in spots.items():
        eb = arm.data.edit_bones.new(name)
        eb.head = pos
        eb.tail = pos + Vector((0, 0, 0.06 if name in [l[0] for l in LIMBS] else 0.035))
        eb.use_deform = False
    bpy.ops.object.mode_set(mode='POSE')
    for handle, pole, *_ in LIMBS:
        _set_color(arm.data.bones[handle], HANDLE_COLOR)
        _set_color(arm.data.bones[pole], POLE_COLOR)
        for name, shape, size in ((handle, HANDLE_SHAPE, HANDLE_SIZE), (pole, POLE_SHAPE, POLE_SIZE)):
            pbone = arm.pose.bones[name]
            pbone.custom_shape = _shape(shape)
            pbone.use_custom_shape_bone_size = False
            pbone.custom_shape_scale_xyz = (size, size, size)
            try:
                pbone.custom_shape_wire_width = 2.5
            except AttributeError:
                pass
        for name in (handle, pole):
            arm.pose.bones[name].lock_rotation = (True, True, True)
            arm.pose.bones[name].lock_rotation_w = True
            arm.pose.bones[name].lock_scale = (True, True, True)


def _remove_handles(arm):
    if not any(n in arm.data.bones for n in HANDLE_NAMES):
        return
    bpy.ops.object.mode_set(mode='EDIT')
    for name in HANDLE_NAMES:
        eb = arm.data.edit_bones.get(name)
        if eb is not None:
            arm.data.edit_bones.remove(eb)
    bpy.ops.object.mode_set(mode='OBJECT')


def _activate(context, arm):
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    arm.hide_set(False)
    for obj in context.view_layer.objects:
        obj.select_set(obj == arm)
    context.view_layer.objects.active = arm


def _rest_rel(arm, name):
    bone = arm.data.bones[name]
    return bone.parent.matrix_local.inverted() @ bone.matrix_local if bone.parent else bone.matrix_local.copy()


def solve(arm) -> bool:
    """ハンドルと目印に合わせて手足を解き直す。姿勢を変えたら True。"""
    pb = arm.pose.bones
    changed = False
    for handle, pole, root, mid, tip, _ in LIMBS:
        if handle not in pb or pole not in pb:
            continue
        H, K, A = pb[root].head.copy(), pb[mid].head.copy(), pb[tip].head.copy()
        T, P = pb[handle].head.copy(), pb[pole].head.copy()
        l1, l2 = (K - H).length, (A - K).length
        to_target = T - H
        d = to_target.length
        if d < 1e-6 or l1 < 1e-6 or l2 < 1e-6:
            continue
        u = to_target / d
        d = min(max(d, abs(l1 - l2) + 1e-5), (l1 + l2) * REACH)
        bend = (P - H) - u * (P - H).dot(u)
        if bend.length < 1e-5:
            bend = (K - H) - u * (K - H).dot(u)
        if bend.length < 1e-5:
            continue
        bend.normalize()
        a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
        h = math.sqrt(max(l1 * l1 - a * a, 0.0))
        K2, T2 = H + u * a + bend * h, H + u * d
        if (K2 - K).length < 1e-4 and (T2 - A).length < 1e-4:
            continue
        r1 = (K - H).rotation_difference(K2 - H)
        r2 = (r1 @ (A - K)).rotation_difference(T2 - K2)
        q_root = r1 @ pb[root].matrix.to_quaternion()
        q_mid = r2 @ r1 @ pb[mid].matrix.to_quaternion()
        q_tip = pb[tip].matrix.to_quaternion()        # 手・足の向きは保つ
        parent = pb[root].parent.matrix.copy() if pb[root].parent else Matrix.Identity(4)
        for name, q in ((root, q_root), (mid, q_mid), (tip, q_tip)):
            rest_m = parent @ _rest_rel(arm, name)
            desired = Matrix.LocRotScale(rest_m.translation, q, None)
            basis = rest_m.inverted() @ desired
            pb[name].rotation_mode = 'QUATERNION'
            pb[name].rotation_quaternion = basis.to_quaternion()
            parent = desired
        changed = True
    return changed


def active_entry(scene):
    mb = getattr(scene, "mb", None)
    if mb is None or not (0 <= mb.edit_index < len(mb.poses)):
        return None
    entry = mb.poses[mb.edit_index]
    return entry if entry.armature is not None else None


def solve_active(scene) -> bool:
    global _solving
    entry = active_entry(scene)
    if entry is None or _solving:
        return False
    _solving = True
    try:
        return solve(entry.armature)
    finally:
        _solving = False


@persistent
def on_depsgraph_update(scene, depsgraph=None):
    if active_entry(scene) is not None:
        solve_active(scene)


def _store_edit(entry):
    """表示中の姿勢と、ポーズ集から取ったままの姿勢の差を保存する。"""
    arm = entry.armature
    bpy.context.view_layer.update()
    now = soma_rig.read_pose(arm, local=True)
    raw = json.loads(entry.window)[poses.display_index(entry)]
    rot = []
    for a, b in zip(now, raw):
        d = Quaternion((a[6], a[3], a[4], a[5])) @ Quaternion((b[6], b[3], b[4], b[5])).inverted()
        rot.append([d.x, d.y, d.z, d.w])
    root = [now[0][k] - raw[0][k] for k in range(3)]
    entry.edited = json.dumps({"rot": rot, "root": root})


def finish(context, scene):
    entry = active_entry(scene)
    mb = scene.mb
    if entry is None:
        mb.edit_index = -1
        return
    arm = entry.armature
    _activate_for_finish(context, arm)
    solve(arm)
    _store_edit(entry)
    mb.edit_index = -1
    _remove_handles(arm)
    poses.show_entry(entry)


def _activate_for_finish(context, arm):
    if context.view_layer.objects.active != arm:
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.objects.active = arm
    if context.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')
    context.view_layer.update()


def finish_if_active(scene):
    """編集中なら確定する（生成や別のポーズの選択の前に呼ぶ）。"""
    if active_entry(scene) is not None:
        finish(bpy.context, scene)


class MB_OT_limb_edit_begin(bpy.types.Operator):
    bl_idname = "mb.limb_edit_begin"
    bl_label = "手足を編集"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        mb = context.scene.mb
        return (mb.edit_index < 0 and not mb.busy and 0 <= mb.index < len(mb.poses)
                and mb.poses[mb.index].armature is not None and bool(mb.poses[mb.index].window))

    def execute(self, context):
        mb = context.scene.mb
        entry = mb.poses[mb.index]
        arm = entry.armature
        _activate(context, arm)
        poses.show_entry(entry)
        context.view_layer.update()
        bpy.ops.object.mode_set(mode='POSE')
        _add_handles(arm)
        context.view_layer.update()
        for pbone in arm.pose.bones:      # Blender 5 は pose bone、古い版は bone 側
            try:
                pbone.select = False
            except AttributeError:
                pbone.bone.select = False
        mb.edit_index = mb.index
        return {'FINISHED'}


class MB_OT_limb_edit_end(bpy.types.Operator):
    bl_idname = "mb.limb_edit_end"
    bl_label = "編集を終える"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return active_entry(context.scene) is not None

    def execute(self, context):
        finish(context, context.scene)
        return {'FINISHED'}


class MB_OT_limb_edit_reset(bpy.types.Operator):
    bl_idname = "mb.limb_edit_reset"
    bl_label = "元に戻す"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        mb = context.scene.mb
        return not mb.busy and 0 <= mb.index < len(mb.poses) and (
            bool(mb.poses[mb.index].edited) or mb.edit_index == mb.index)

    def execute(self, context):
        scene, mb = context.scene, context.scene.mb
        entry = mb.poses[mb.index]
        if mb.edit_index == mb.index:
            arm = entry.armature
            _activate_for_finish(context, arm)
            mb.edit_index = -1
            _remove_handles(arm)
        entry.edited = ""
        poses.show_entry(entry)
        return {'FINISHED'}


classes = (MB_OT_limb_edit_begin, MB_OT_limb_edit_end, MB_OT_limb_edit_reset)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)


def unregister():
    if on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
