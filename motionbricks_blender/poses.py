"""ポーズ一覧（最初・途中・最後）とポーズ人形。

人形（SOMA 骨格のアーマチュア）は床の上に置く。人形の位置と向きがポーズの配置で、
姿勢は人形のローカル座標（原点に立ち、-Y を向く）で持つ。だから人形をドラッグ
しても取り直しは要らない。
"""

import json
import math

import bpy
from bpy.app.handlers import persistent
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)
from mathutils import Matrix, Quaternion

from . import library, soma_rig
from .client import Client, ClientError

COLLECTION = "MB_ポーズ"
COLORS = {'START': (0.24, 0.81, 0.38), 'MIDDLE': (0.88, 0.63, 0.23), 'END': (0.31, 0.56, 0.91)}
LOCAL_PLACEMENT = [0.0, 0.0, -math.pi / 2]   # 原点に立ち、-Y を向く

_client_factory = None
_muted = False
_ENUM_ITEMS = [(i["id"], i["name"], "") for i in library.items()] or [("base", "基本姿勢", "")]


def library_item(item_id):
    return library.item(item_id)


def set_client_factory(factory):
    """テスト用：サーバーの代わりを差し込む。None で元に戻す。"""
    global _client_factory
    _client_factory = factory


def get_client(scene=None):
    if _client_factory is not None:
        return _client_factory()
    scene = scene or bpy.context.scene
    mb = scene.mb
    if mb.status == 'RUNNING' and mb.server_port:
        return Client(mb.server_port)
    return None


# ---- ポーズの計算 -----------------------------------------------------------

def _q(t):
    return Quaternion((t[6], t[3], t[4], t[5]))


def apply_edit(window, edit):
    """手足の編集（関節ごとの回転の差と腰の移動）を 4 ポーズすべてに適用する。"""
    if not edit:
        return window
    out = []
    for pose in window:
        new = []
        for j, t in enumerate(pose):
            d = edit["rot"][j]
            q = Quaternion((d[3], d[0], d[1], d[2])) @ _q(t)
            p = list(t[:3])
            if j == 0:
                p = [p[k] + edit["root"][k] for k in range(3)]
            new.append(p + [q.x, q.y, q.z, q.w])
        out.append(new)
    return out


def display_index(entry):
    """人形に表示するポーズ：最初は 4 ポーズの最後、ほかは最初。"""
    return 3 if entry.kind == 'START' else 0


def local_window(entry):
    """人形ローカルの 4 ポーズ（編集を適用済み）。"""
    if not entry.window:
        return None
    edit = json.loads(entry.edited) if entry.edited else None
    return apply_edit(json.loads(entry.window), edit)


def placement_matrix(arm):
    return (Matrix.Translation((arm.location.x, arm.location.y, 0.0))
            @ Matrix.Rotation(arm.rotation_euler.z, 4, 'Z'))


def entry_window(entry):
    """ワールドでの 4 ポーズ（生成に渡す形）。"""
    window = local_window(entry)
    if window is None or entry.armature is None:
        return None
    place = placement_matrix(entry.armature)
    out = []
    for pose in window:
        root = place @ soma_rig._mat(pose[0])
        out.append([soma_rig._xform(root)] + [list(t) for t in pose[1:]])
    return out


def show_entry(entry):
    window = local_window(entry)
    if window is None or entry.armature is None:
        return
    soma_rig.apply_pose(entry.armature, window[display_index(entry)], local=True)


def refresh_pose(entry, client=None):
    """ポーズ集から取り直して人形に表示する。"""
    client = client or get_client()
    if client is None or entry.armature is None:
        return False
    item = library.item(entry.library_id)
    poses = client.sample(item["file"], entry.time * item["duration"], display_index(entry),
                          LOCAL_PLACEMENT, static=bool(item.get("static")))
    entry.window = json.dumps(poses)
    show_entry(entry)
    return True


# ---- プロパティ -------------------------------------------------------------

def _on_library_changed(self, context):
    global _muted
    if _muted:
        return
    item = library.item(self.library_id)
    _muted = True
    try:
        self.time = item["default_time"] / item["duration"] if item["duration"] else 0.0
    finally:
        _muted = False
    _safe_refresh(self, context)


def _on_time_changed(self, context):
    if not _muted:
        _safe_refresh(self, context)


def _safe_refresh(entry, context):
    try:
        refresh_pose(entry)
    except ClientError as e:
        context.scene.mb.error = str(e)


def _on_index_changed(self, context):
    if not (0 <= self.index < len(self.poses)):
        return
    arm = self.poses[self.index].armature
    if arm is None or arm.name not in context.view_layer.objects:
        return
    try:
        if context.mode != 'OBJECT':
            return
        arm.hide_set(False)
        for obj in context.view_layer.objects:
            obj.select_set(obj == arm)
        context.view_layer.objects.active = arm
    except RuntimeError:
        pass


class MBPoseEntry(bpy.types.PropertyGroup):
    kind: EnumProperty(items=[('START', "最初", ""), ('MIDDLE', "途中", ""), ('END', "最後", "")])
    library_id: EnumProperty(name="ポーズ集", items=_ENUM_ITEMS, update=_on_library_changed)
    time: FloatProperty(name="タイミング", min=0.0, max=1.0, subtype='FACTOR', update=_on_time_changed)
    armature: PointerProperty(type=bpy.types.Object)
    window: StringProperty()
    edited: StringProperty()


class MBSceneState(bpy.types.PropertyGroup):
    poses: CollectionProperty(type=MBPoseEntry)
    index: IntProperty(update=_on_index_changed)
    edit_index: IntProperty(default=-1)
    skeleton: StringProperty()
    server_port: IntProperty()
    status: EnumProperty(items=[('STOPPED', "停止", ""), ('INSTALLING', "準備中", ""), ('STARTING', "起動中", ""),
                                ('RUNNING', "起動中", ""), ('ERROR', "エラー", "")], default='STOPPED')
    progress: FloatProperty(min=0.0, max=1.0, subtype='FACTOR')
    progress_label: StringProperty()
    error: StringProperty()
    busy: BoolProperty()
    character_path: StringProperty(name="キャラクター", subtype='FILE_PATH')
    export_mode: EnumProperty(items=[('BAKE', "キャラに焼き込み", ""), ('MOTION', "動きのみ", "")], default='BAKE')
    in_place: BoolProperty(name="In Place")
    variant: IntProperty()
    result_action: PointerProperty(type=bpy.types.Action)


# ---- 人形 ------------------------------------------------------------------

def entry_label(mb, i):
    kind = mb.poses[i].kind
    if kind == 'START':
        return "最初"
    if kind == 'END':
        return "最後"
    return f"途中 {sum(1 for e in list(mb.poses)[:i + 1] if e.kind == 'MIDDLE')}"


def pose_collection(scene):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def _rename(mb):
    for i, entry in enumerate(mb.poses):
        if entry.armature is not None:
            entry.armature.name = "MB_" + entry_label(mb, i).replace(" ", "")


def _make_puppet(scene, entry, location, rot_z):
    skel = json.loads(scene.mb.skeleton)
    arm = soma_rig.build_armature(skel, "MB_ポーズ", COLORS[entry.kind], pose_collection(scene))
    arm.location = (location[0], location[1], 0.0)
    arm.rotation_euler = (0.0, 0.0, rot_z)
    arm.lock_location = (False, False, True)
    arm.lock_rotation = (True, True, False)
    arm.lock_scale = (True, True, True)
    entry.armature = arm
    return arm


def _add_entry(scene, kind, library_id, time, location, rot_z):
    global _muted
    mb = scene.mb
    entry = mb.poses.add()
    _muted = True
    try:
        entry.kind = kind
        entry.library_id = library_id
        item = library.item(library_id)
        entry.time = time if time is not None else (item["default_time"] / item["duration"])
    finally:
        _muted = False
    _make_puppet(scene, entry, location, rot_z)
    return entry


def on_server_started(scene, skel):
    """起動直後：骨格を覚え、ポーズが無ければ最初と最後を作る。"""
    mb = scene.mb
    mb.skeleton = json.dumps(skel)
    mb.status = 'RUNNING'
    mb.error = ""
    if len(mb.poses) == 0:
        _add_entry(scene, 'START', "stand", None, (0.0, 0.0), 0.0)
        _add_entry(scene, 'END', "stand", None, (0.0, -2.0), 0.0)
        mb.index = 0
    for entry in mb.poses:
        if entry.armature is None:
            _make_puppet(scene, entry, (0.0, 0.0), 0.0)
        if not entry.window:
            refresh_pose(entry)
        else:
            show_entry(entry)
    _rename(mb)


class MB_OT_pose_add(bpy.types.Operator):
    bl_idname = "mb.pose_add"
    bl_label = "途中を追加"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.scene.mb.poses) >= 2 and not context.scene.mb.busy

    def execute(self, context):
        scene, mb = context.scene, context.scene.mb
        sel = min(max(mb.index, 0), len(mb.poses) - 1)
        pos = sel if mb.poses[sel].kind == 'END' else sel + 1
        prev_arm, next_arm = mb.poses[pos - 1].armature, mb.poses[pos].armature
        loc = (prev_arm.location.xy + next_arm.location.xy) / 2
        rot_z = prev_arm.rotation_euler.z
        source = mb.poses[sel]
        from_start = source.kind == 'START'
        lib, time, window, edited = source.library_id, source.time, source.window, source.edited
        entry = _add_entry(scene, 'MIDDLE', lib, time, loc, rot_z)
        if from_start or not window:
            try:
                refresh_pose(entry)       # 最初のポーズは取り方が違うので取り直す
            except ClientError as e:
                mb.error = str(e)
        else:
            entry.window, entry.edited = window, edited
            show_entry(entry)
        mb.poses.move(len(mb.poses) - 1, pos)
        _rename(mb)
        mb.index = pos
        return {'FINISHED'}


class MB_OT_pose_remove(bpy.types.Operator):
    bl_idname = "mb.pose_remove"
    bl_label = "削除"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        mb = context.scene.mb
        return 0 <= mb.index < len(mb.poses) and mb.poses[mb.index].kind == 'MIDDLE' and not mb.busy

    def execute(self, context):
        mb = context.scene.mb
        arm = mb.poses[mb.index].armature
        mb.poses.remove(mb.index)
        if arm is not None:
            data = arm.data
            bpy.data.objects.remove(arm, do_unlink=True)
            if data.users == 0:
                bpy.data.armatures.remove(data)
        _rename(mb)
        mb.index = min(mb.index, len(mb.poses) - 1)
        return {'FINISHED'}


class MB_OT_pose_move(bpy.types.Operator):
    bl_idname = "mb.pose_move"
    bl_label = "並べ替え"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(items=[('UP', "上へ", ""), ('DOWN', "下へ", "")])

    @classmethod
    def poll(cls, context):
        return MB_OT_pose_remove.poll(context)

    def execute(self, context):
        mb = context.scene.mb
        other = mb.index + (-1 if self.direction == 'UP' else 1)
        if not (0 <= other < len(mb.poses)) or mb.poses[other].kind != 'MIDDLE':
            return {'CANCELLED'}
        mb.poses.move(mb.index, other)
        _rename(mb)
        mb.index = other
        return {'FINISHED'}


@persistent
def on_load_post(*_):
    """ファイルを開いた直後はサーバーが無いので、状態を「停止」に戻す。"""
    for scene in bpy.data.scenes:
        mb = getattr(scene, "mb", None)
        if mb is not None:
            mb.status = 'STOPPED'
            mb.progress = 0.0
            mb.error = ""
            mb.busy = False


classes = (MBPoseEntry, MBSceneState, MB_OT_pose_add, MB_OT_pose_remove, MB_OT_pose_move)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mb = PointerProperty(type=MBSceneState)
    bpy.app.handlers.load_post.append(on_load_post)


def unregister():
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)
    del bpy.types.Scene.mb
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
