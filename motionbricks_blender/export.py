"""Mixamo キャラクターの読み込みと FBX の書き出し。"""

import json
import os
import time

import bpy
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix

from . import rig_transfer as rt

CHAR_COLLECTION = "MB_キャラクター"
PATH_KEY = "mb_char_path"
MTIME_KEY = "mb_char_mtime"


def _collection(scene):
    coll = bpy.data.collections.get(CHAR_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(CHAR_COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def character_armature(scene):
    coll = bpy.data.collections.get(CHAR_COLLECTION)
    if coll is None:
        return None
    return next((o for o in coll.objects if o.type == 'ARMATURE' and PATH_KEY in o), None)


def character_objects(scene):
    coll = bpy.data.collections.get(CHAR_COLLECTION)
    return list(coll.objects) if coll else []


def _delete(objects):
    datas = [o.data for o in objects if o.data is not None]
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for data in datas:
        if data.users == 0:
            try:
                bpy.data.batch_remove([data])
            except Exception:
                pass


def remove_character(scene):
    _delete(character_objects(scene))


def ensure_character(scene, path: str):
    """Mixamo キャラクターの FBX を読み込む（同じファイルなら読み直さない）。元のファイルは変更しない。"""
    full = os.path.normpath(bpy.path.abspath(path))
    if not os.path.isfile(full):
        raise ValueError("キャラクターのファイルが見つかりません")
    mtime = os.path.getmtime(full)
    existing = character_armature(scene)
    if existing is not None and existing[PATH_KEY] == full and abs(existing[MTIME_KEY] - mtime) < 1e-3:
        return existing

    context = bpy.context
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    view_layer = context.view_layer
    prev_active = view_layer.objects.active
    prev_selected = [o for o in view_layer.objects if o.select_get()]
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.fbx(filepath=full)
    except Exception as e:
        _delete([o for o in bpy.data.objects if o not in before])
        raise ValueError("キャラクターを読み込めませんでした") from e
    new = [o for o in bpy.data.objects if o not in before]
    arm = next((o for o in new if o.type == 'ARMATURE' and rt.detect_mixamo_prefix(o) is not None), None)
    if arm is None:
        _delete(new)
        raise rt.NotMixamoError("Mixamoのキャラクターではありません")

    remove_character(scene)
    coll = _collection(scene)
    for obj in new:
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        coll.objects.link(obj)
        obj.select_set(False)
    arm[PATH_KEY] = full
    arm[MTIME_KEY] = mtime
    for obj in prev_selected:
        try:
            obj.select_set(True)
        except (RuntimeError, ReferenceError):
            pass
    try:
        view_layer.objects.active = prev_active
    except ReferenceError:
        pass
    return arm


RESULT_KEY = "mb_result"


def ensure_character_motion(scene):
    """キャラクターを読み込み、いまの生成結果を転送する（転送済みなら使い回す）。(キャラ, アクション)。"""
    from . import foot_lock, generate
    mb = scene.mb
    result = mb.result_action
    source = generate.result_armature(scene)
    if result is None or source is None:
        raise ValueError("先に生成してください")
    char = ensure_character(scene, mb.character_path)
    current = char.animation_data.action if char.animation_data else None
    if current is not None and current.get(RESULT_KEY) == result.name:
        generate.sync_playback_in_place(scene)
        return char, current
    rt.bind_action(source, result)
    # 接地補正は移動を含む動きで計算し、再生用の In Place は転送後に反映する。
    was_in_place = rt.is_in_place(source)
    ok, message = rt.remove_in_place(source)
    if not ok:
        raise ValueError(f"元の移動を復元できません: {message}")
    try:
        action = rt.transfer_action(source, char, rt.unique_action_name("MB_Mixamo"))
        foot_lock.lock_action(char, action, json.loads(result["mb_contacts"]), start=int(result.frame_range[0]))
    finally:
        if was_in_place:
            ok, message = rt.apply_in_place(source)
            if not ok:
                raise ValueError(f"In Placeを復元できません: {message}")
    action[RESULT_KEY] = result.name
    generate.sync_playback_in_place(scene)
    return char, action


def _normalise_start(scene, char, action, frames):
    """最初のフレームで腰が原点、最初のポーズの正面が -Y になるように、動き全体を動かす。"""
    prefix = rt.detect_mixamo_prefix(char)
    hips = prefix + "Hips"
    bone = char.data.bones[hips]
    sampler = rt.PoseSampler(char, action)
    mw = char.matrix_world
    parent_rest = bone.parent.matrix_local if bone.parent else Matrix.Identity(4)
    rest = mw @ bone.matrix_local
    first = (rest @ (bone.matrix_local.inverted() @ parent_rest @ rt.rest_relative(char)[hips]
                     @ sampler.basis(hips, frames[0]))).translation
    start = scene.mb.poses[0].armature if len(scene.mb.poses) else None
    turn = -start.rotation_euler.z if start is not None else 0.0
    move = Matrix.Rotation(turn, 4, 'Z') @ Matrix.Translation((-first.x, -first.y, 0.0))
    local = parent_rest @ rt.rest_relative(char)[hips]          # 腰の基本行列（アーマチュア座標）
    fix = local.inverted() @ mw.inverted() @ move @ mw @ local
    bases = [{hips: fix @ sampler.basis(hips, f)} for f in frames]
    rt.write_bases(char, action, frames, bases, [hips], root_bones=(hips,))


def export_fbx(context, scene, filepath: str):
    """生成結果を Mixamo 形式の FBX に書き出す。キャラクターと元のアクションは変更しない。"""
    mb = scene.mb
    if not mb.character_path:
        raise ValueError("キャラクターを指定してください")
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    char, action = ensure_character_motion(scene)
    filepath = bpy.path.ensure_ext(bpy.path.abspath(filepath), ".fbx")
    frames = list(range(int(action.frame_range[0]), int(action.frame_range[1]) + 1))

    view_layer = context.view_layer
    prev_active = view_layer.objects.active
    prev_selected = [o for o in view_layer.objects if o.select_get()]
    prev_range = (scene.frame_start, scene.frame_end, scene.frame_current)
    prev_scene_name = scene.name
    members = [char] + ([o for o in character_objects(scene) if o.type == 'MESH']
                        if mb.export_mode == 'BAKE' else [])
    hidden = [(o, o.hide_get(), o.hide_viewport) for o in members]
    temp = action.copy()
    temp.name = "MB_書き出し"
    try:
        rt.bind_action(char, temp)
        ok, message = rt.remove_in_place(char)
        if not ok:
            raise ValueError(f"元の移動を復元できません: {message}")
        _normalise_start(scene, char, temp, frames)
        if mb.in_place:
            ok, message = rt.apply_in_place(char)
            if not ok:
                raise ValueError("In Place にできませんでした")
        # FBX の時刻が 0 秒から始まるように、0 フレーム始まりにずらす（Mixamo の FBX と同じ）
        for fc in rt.action_fcurves(temp) or []:
            for kp in fc.keyframe_points:
                kp.co.x -= frames[0]
                kp.handle_left.x -= frames[0]
                kp.handle_right.x -= frames[0]
            fc.update()
        scene.frame_start, scene.frame_end = 0, frames[-1] - frames[0]
        # FBX のクリップ名にはシーン名が使われる。ファイル名と同じにしておく
        scene.name = os.path.splitext(os.path.basename(filepath))[0] or "MotionBricks"
        scene.render.fps, scene.render.fps_base = 30, 1.0
        for obj in view_layer.objects:
            obj.select_set(False)
        for obj in members:
            obj.hide_viewport = False
            obj.hide_set(False)
            obj.select_set(True)
        view_layer.objects.active = char
        bpy.ops.export_scene.fbx(
            filepath=filepath, use_selection=True,
            object_types={'ARMATURE', 'MESH'} if mb.export_mode == 'BAKE' else {'ARMATURE'},
            bake_anim=True, bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
            bake_anim_force_startend_keying=True, bake_anim_simplify_factor=0.0,
            add_leaf_bones=False, primary_bone_axis='Y', secondary_bone_axis='X',
            axis_forward='-Z', axis_up='Y', apply_scale_options='FBX_SCALE_ALL',
        )
    finally:
        scene.name = prev_scene_name
        rt.bind_action(char, action)
        bpy.data.actions.remove(temp)
        for obj, was_hidden, was_disabled in hidden:
            obj.hide_viewport = was_disabled
            obj.hide_set(was_hidden)
        for obj in view_layer.objects:
            obj.select_set(obj in prev_selected)
        view_layer.objects.active = prev_active
        scene.frame_start, scene.frame_end = prev_range[0], prev_range[1]
        scene.frame_set(prev_range[2])
    return filepath


class MB_OT_export(bpy.types.Operator, ExportHelper):
    bl_idname = "mb.export"
    bl_label = "Mixamo形式で保存"
    bl_options = {'REGISTER'}

    filename_ext = ".fbx"
    filter_glob: bpy.props.StringProperty(default="*.fbx", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        mb = context.scene.mb
        return mb.result_action is not None and bool(mb.character_path) and not mb.busy

    def invoke(self, context, event):
        self.filepath = time.strftime("MB_%Y%m%d_%H%M%S.fbx")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        mb = context.scene.mb
        try:
            path = export_fbx(context, context.scene, self.filepath)
        except ValueError as e:
            mb.error = str(e)
            self.report({'ERROR'}, mb.error)
            return {'CANCELLED'}
        mb.error = ""
        self.report({'INFO'}, f"保存しました: {os.path.basename(path)}")
        return {'FINISHED'}


classes = (MB_OT_export,)



def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
