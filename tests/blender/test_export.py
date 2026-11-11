import math
import os
import tempfile

import bpy
from mathutils import Vector

import common
from motionbricks_blender import export, generate, poses, rig_transfer as rt

P = "mixamorig:"
CHECK = ("Hips", "Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot", "LeftLeg", "RightForeArm")


def teardown():
    poses.set_client_factory(None)


def teardown_module():
    common.stop_real()


def _mb():
    return bpy.context.scene.mb


def _generated(walk_far=True, start_turn=0.0):
    common.start_real()
    mb = _mb()
    if walk_far:
        mb.poses[1].armature.location = (0.6, -3.0, 0.0)
    mb.poses[0].armature.rotation_euler.z = start_turn
    mb.character_path = str(common.character_path())
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error
    return mb


def _heads(arm, frames, prefix=P):
    scene = bpy.context.scene
    out = {}
    for f in frames:
        scene.frame_set(f)
        bpy.context.view_layer.update()
        out[f] = {n: (arm.matrix_world @ arm.pose.bones[prefix + n].head).copy() for n in CHECK}
    return out


def _frames():
    end = int(_mb().result_action.frame_range[1])
    return list(range(1, end + 1, 10)) + [end]


def _out(name="out.fbx", sub="書き出し テスト"):
    folder = os.path.join(tempfile.mkdtemp(), sub)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def _reimport(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.fbx(filepath=path)
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    return arm, meshes


def _expected(before, frames):
    """書き出しでは、最初のフレームの腰が原点に来るように平行移動する。"""
    shift = Vector((before[frames[0]]["Hips"].x, before[frames[0]]["Hips"].y, 0.0))
    return {f: {n: p - shift for n, p in before[f].items()} for f in frames}


def test_bake_export_reimport():
    mb = _generated()
    char = generate.character_armature(bpy.context.scene)
    frames = _frames()
    want = _expected(_heads(char, frames), frames)
    path = _out("キャラ.fbx")
    mb.export_mode = 'BAKE'
    assert bpy.ops.mb.export(filepath=path) == {'FINISHED'}, mb.error
    assert os.path.getsize(path) > 100000
    arm, meshes = _reimport(path)
    assert len(meshes) >= 1 and rt.detect_mixamo_prefix(arm) == P
    assert int(arm.animation_data.action.frame_range[1]) == frames[-1], (tuple(arm.animation_data.action.frame_range), frames[-1], bpy.context.scene.render.fps)
    got = _heads(arm, frames)
    worst = max((got[f][n] - want[f][n]).length for f in frames for n in CHECK)
    assert worst < 1e-3, worst


def test_motion_only_has_no_mesh():
    mb = _generated()
    char = generate.character_armature(bpy.context.scene)
    frames = _frames()
    want = _expected(_heads(char, frames), frames)
    path = _out("動きのみ.fbx")
    mb.export_mode = 'MOTION'
    assert bpy.ops.mb.export(filepath=path) == {'FINISHED'}, mb.error
    arm, meshes = _reimport(path)
    assert meshes == [] and rt.detect_mixamo_prefix(arm) == P
    got = _heads(arm, frames)
    worst = max(((got[f][n] - want[f][n]).length, f, n) for f in frames for n in CHECK)
    assert worst[0] < 1e-3, (worst, got[worst[1]][worst[2]], want[worst[1]][worst[2]], arm.scale[:])


def test_starts_at_origin_and_in_place():
    mb = _generated()
    char = generate.character_armature(bpy.context.scene)
    frames = _frames()
    before = _heads(char, frames)
    assert (before[frames[-1]]["Hips"].xy - before[1]["Hips"].xy).length > 1.0
    mb.in_place = True
    path = _out("その場.fbx")
    assert bpy.ops.mb.export(filepath=path) == {'FINISHED'}, mb.error
    arm, _ = _reimport(path)
    got = _heads(arm, frames)
    for f in frames:
        assert got[f]["Hips"].xy.length < 1e-3, (f, got[f]["Hips"])              # 水平移動なし、原点
        assert abs(got[f]["Hips"].z - before[f]["Hips"].z) < 1e-3                 # 高さはそのまま


def test_start_facing_is_normalised():
    """最初のポーズを回して置いても、書き出した動きは正面（-Y）向きで始まる。"""
    results = []
    for turn in (0.0, 1.1):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mb = _generated(walk_far=False, start_turn=turn)
        path = _out(f"turn{turn}.fbx")
        assert bpy.ops.mb.export(filepath=path) == {'FINISHED'}, mb.error
        arm, _ = _reimport(path)
        bpy.context.scene.frame_set(1)
        bpy.context.view_layer.update()
        left = arm.matrix_world @ arm.pose.bones[P + "LeftUpLeg"].head
        right = arm.matrix_world @ arm.pose.bones[P + "RightUpLeg"].head
        results.append(math.atan2((right - left).y, (right - left).x))
        assert (arm.matrix_world @ arm.pose.bones[P + "Hips"].head).xy.length < 1e-3
    assert abs(results[0] - results[1]) < 0.02, results
    assert abs(abs(results[0]) - math.pi) < 0.2, results          # -Y を向くと、右の股関節は -X 側


def test_source_untouched():
    mb = _generated()
    scene = bpy.context.scene
    char = generate.character_armature(scene)
    src = str(common.character_path())
    stamp = (os.path.getmtime(src), os.path.getsize(src))
    result_sig = common.action_signature(mb.result_action)
    char_action = char.animation_data.action
    char_sig = common.action_signature(char_action)
    n_actions = len(bpy.data.actions)
    selected = [o.name for o in bpy.context.view_layer.objects if o.select_get()]
    mb.in_place = True
    assert bpy.ops.mb.export(filepath=_out()) == {'FINISHED'}, mb.error
    assert (os.path.getmtime(src), os.path.getsize(src)) == stamp
    assert common.action_signature(mb.result_action) == result_sig
    assert char.animation_data.action == char_action and common.action_signature(char_action) == char_sig
    assert len(bpy.data.actions) == n_actions                                   # 作業用のアクションを残さない
    assert [o.name for o in bpy.context.view_layer.objects if o.select_get()] == selected


def test_character_set_after_generating():
    common.start_real()
    mb = _mb()
    assert bpy.ops.mb.generate() == {'FINISHED'}
    assert bpy.ops.mb.export.poll() is False                                    # キャラクター未指定
    mb.character_path = str(common.character_path())
    assert bpy.ops.mb.export.poll() is True
    path = _out("後から.fbx")
    assert bpy.ops.mb.export(filepath=path) == {'FINISHED'}, mb.error
    arm, meshes = _reimport(path)
    assert rt.detect_mixamo_prefix(arm) == P and int(arm.animation_data.action.frame_range[1]) > 20


def test_adds_fbx_extension():
    mb = _generated(walk_far=False)
    path = _out("名前だけ")
    assert bpy.ops.mb.export(filepath=path) == {'FINISHED'}, mb.error
    assert os.path.isfile(path + ".fbx")


def test_not_mixamo_cleanup():
    common.start_real()
    mb = _mb()
    bpy.ops.mb.generate()
    scene = bpy.context.scene
    arm = generate.result_armature(scene)
    arm.hide_set(False)
    for o in bpy.context.view_layer.objects:
        o.select_set(o == arm)
    bpy.context.view_layer.objects.active = arm
    bad = _out("ちがう骨格.fbx")
    bpy.ops.export_scene.fbx(filepath=bad, use_selection=True, object_types={'ARMATURE'}, bake_anim=False)
    n = len(bpy.data.objects)
    mb.character_path = bad
    try:
        result = bpy.ops.mb.export(filepath=_out("x.fbx"))
    except RuntimeError:
        result = {'CANCELLED'}
    assert result == {'CANCELLED'}
    assert "Mixamo" in mb.error and any(ord(ch) > 0x3000 for ch in mb.error)
    assert len(bpy.data.objects) == n
    assert export.character_armature(scene) is None


def test_clip_is_named_after_the_file():
    """ゲームエンジンで見えるクリップ名は、保存したファイル名にする（既定だと "Scene" になる）。"""
    mb = _generated(walk_far=False)
    scene_name = bpy.context.scene.name
    path = _out("Hero_Wave.fbx")
    assert bpy.ops.mb.export(filepath=path) == {'FINISHED'}, mb.error
    assert bpy.context.scene.name == scene_name                 # シーン名は元に戻す
    data = open(path, "rb").read()
    assert b"Hero_Wave" in data and b"AnimStack::Scene" not in data
    arm, _ = _reimport(path)
    assert "Hero_Wave" in arm.animation_data.action.name
