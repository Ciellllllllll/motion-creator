"""生成・再生・もう一度。"""

import json

import bpy

from . import chain, export, foot_lock, library, limb_edit, poses, soma_rig
from . import rig_transfer as rt
from .client import ClientError

RESULT_NAME = "MB_結果"
RESULT_COLLECTION = "MB_結果"
RESULT_COLOR = (0.9, 0.9, 0.9)
FPS = 30


def result_armature(scene, create=False):
    arm = bpy.data.objects.get(RESULT_NAME)
    if arm is not None and arm.type == 'ARMATURE' and soma_rig.SKEL_KEY in arm:
        return arm
    if not create:
        return None
    coll = bpy.data.collections.get(RESULT_COLLECTION) or bpy.data.collections.new(RESULT_COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return soma_rig.build_armature(json.loads(scene.mb.skeleton), RESULT_NAME, RESULT_COLOR, coll)


def character_armature(scene):
    return export.character_armature(scene)


def _is_connection_error(message):
    return "接続" in message or "応答" in message


def run(context, scene):
    """いまのポーズから動きを生成して表示する。失敗は ClientError / ValueError。"""
    mb = scene.mb
    limb_edit.finish_if_active(scene)
    client = poses.get_client(scene)
    if client is None:
        raise ClientError("MotionBricksを起動してください")
    entries = list(mb.poses)
    windows = [poses.entry_window(e) for e in entries]
    if len(windows) < 2 or any(w is None for w in windows):
        raise ValueError("ポーズが足りません")
    walk = library.item("walk")

    def walk_sampler(x, y, yaw):
        return client.sample(walk["file"], walk["default_time"], 0, [x, y, yaw])

    request = chain.build_request(windows, [e.kind for e in entries], walk_sampler, mb.variant)
    reply = client.generate(request["context"], request["targets"], request["token_counts"])

    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    arm = result_armature(scene, create=True)
    arm.hide_set(False)
    action = soma_rig.write_action(arm, reply["frames"], rt.unique_action_name("MB_生成"), start=1)
    action["mb_pose_frames"] = json.dumps([reply["segments"][i][1] + 1 for i in request["pose_targets"]])
    action["mb_contacts"] = json.dumps(reply["contacts"])
    mb.result_action = action

    warning = ""
    if mb.character_path:
        try:
            export.ensure_character_motion(scene)
            arm.hide_set(True)
        except ValueError as e:        # NotMixamoError を含む。生成そのものは成功
            warning = str(e)

    for entry in entries:
        if entry.armature is not None:
            entry.armature.hide_set(True)
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = len(reply["frames"])
    scene.frame_set(1)
    return warning


def _start_playback(context):
    if bpy.app.background or context.screen is None:
        return
    if not context.screen.is_animation_playing:
        try:
            bpy.ops.screen.animation_play()
        except RuntimeError:
            pass


class MB_OT_generate(bpy.types.Operator):
    bl_idname = "mb.generate"
    bl_label = "生成"
    bl_options = {'REGISTER', 'UNDO'}

    keep_variant: bpy.props.BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        mb = context.scene.mb
        return len(mb.poses) >= 2 and not mb.busy

    def execute(self, context):
        scene, mb = context.scene, context.scene.mb
        if not self.keep_variant:
            mb.variant = 0
        try:
            warning = run(context, scene)
        except ClientError as e:
            mb.error = str(e)
            if _is_connection_error(mb.error):
                mb.status = 'ERROR'
            self.report({'ERROR'}, mb.error)
            return {'CANCELLED'}
        except ValueError as e:
            mb.error = str(e)
            self.report({'ERROR'}, mb.error)
            return {'CANCELLED'}
        mb.error = warning
        if warning:
            self.report({'WARNING'}, warning)
        _start_playback(context)
        return {'FINISHED'}


class MB_OT_regenerate(bpy.types.Operator):
    bl_idname = "mb.regenerate"
    bl_label = "もう一度"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return MB_OT_generate.poll(context) and context.scene.mb.result_action is not None

    def execute(self, context):
        mb = context.scene.mb
        before = int(mb.result_action.frame_range[1])
        result = {'CANCELLED'}
        for _ in range(3):      # 前と同じ長さになったら、次の候補でやり直す
            mb.variant += 1
            result = bpy.ops.mb.generate(keep_variant=True)
            if result != {'FINISHED'} or int(mb.result_action.frame_range[1]) != before:
                break
        return result


class MB_OT_play(bpy.types.Operator):
    bl_idname = "mb.play"
    bl_label = "再生"

    @classmethod
    def poll(cls, context):
        return context.scene.mb.result_action is not None

    def execute(self, context):
        screen = context.screen
        if screen is None:
            return {'CANCELLED'}
        if screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        else:
            context.scene.frame_set(context.scene.frame_start)
            bpy.ops.screen.animation_play()
        return {'FINISHED'}


classes = (MB_OT_generate, MB_OT_regenerate, MB_OT_play)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
