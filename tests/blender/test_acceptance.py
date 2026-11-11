"""設計書 9 章の完成の条件を、まっさらな状態から通しで確認する。

MOTIONBRICKS_BLENDER_HOME を新しいフォルダ（日本語を含む）にして、インストールから
書き出しまでを行う。初回は数 GB の取得と TensorRT の準備で数分かかる。
"""
import os
import tempfile
import time

import bpy
from mathutils import Vector

import common
from motionbricks_blender import generate, poses, rig_transfer as rt, setup_env, ui

P = "mixamorig:"
_state = {}


def teardown_module():
    ui.stop_server()


def _reimport(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.fbx(filepath=path)
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    return arm, [o for o in bpy.data.objects if o.type == 'MESH']


def test_everything_from_a_clean_machine():
    home = os.path.join(tempfile.mkdtemp(prefix="mb_"), "MB 受け入れ")
    log = os.path.join(tempfile.mkdtemp(prefix="mb_"), "access.log")
    open(log, "w").close()
    os.environ["MOTIONBRICKS_BLENDER_HOME"] = home
    os.environ["MB_TRACK_ACCESS"] = log
    assert not setup_env.is_installed()
    scene = bpy.context.scene
    mb = scene.mb

    # 条件 1：「起動」を押すだけでインストールから起動まで進む
    t0 = time.time()
    assert bpy.ops.mb.start() == {'FINISHED'}, mb.error
    print(f"ACCEPT first launch {time.time() - t0:.0f} s", flush=True)
    assert mb.status == 'RUNNING' and setup_env.is_installed()
    assert os.path.isdir(os.path.join(home, "motionbricks", ".venv"))

    # 条件 2・3：最初・途中 2・最後（1 つは基本姿勢から手足を編集）で生成
    mb.index = 0
    bpy.ops.mb.pose_add()
    bpy.ops.mb.pose_add()
    mb.poses[1].library_id = "walk"
    mb.poses[1].armature.location = (0.0, -1.5, 0.0)
    mb.poses[2].library_id = "base"
    mb.poses[2].armature.location = (0.5, -3.0, 0.0)
    mb.poses[3].library_id = "wave"
    mb.poses[3].armature.location = (0.5, -4.5, 0.0)
    mb.index = 2
    bpy.ops.mb.limb_edit_begin()
    arm = mb.poses[2].armature
    pb = arm.pose.bones["MB_右手"]
    pb.location = pb.location + arm.data.bones["MB_右手"].matrix_local.to_3x3().inverted() @ Vector((0, -0.2, 0.5))
    bpy.context.view_layer.update()
    src = str(common.character_path())
    stamp = (os.path.getmtime(src), os.path.getsize(src))
    mb.character_path = src
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error
    import json
    frames = json.loads(mb.result_action["mb_pose_frames"])
    result = generate.result_armature(scene)
    for entry, frame in zip(list(mb.poses)[1:], frames):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        got = (result.matrix_world @ result.pose.bones["Hips"].head).xy
        want = Vector(poses.entry_window(entry)[0][0][:2])
        assert (got - want).length < 0.10, (frame, got, want)

    # 条件 4・5：両方の形式で書き出し、読み直せる。元のファイルは変わらない
    out = os.path.join(tempfile.mkdtemp(prefix="mb_"), "書き出し")
    os.makedirs(out)
    paths = {}
    for mode, in_place in (('BAKE', False), ('MOTION', True)):
        mb.export_mode, mb.in_place = mode, in_place
        paths[mode] = os.path.join(out, f"{mode}.fbx")
        assert bpy.ops.mb.export(filepath=paths[mode]) == {'FINISHED'}, mb.error
    assert (os.path.getmtime(src), os.path.getsize(src)) == stamp
    end = int(mb.result_action.frame_range[1])
    _state["paths"] = paths

    arm, meshes = _reimport(paths['BAKE'])
    assert meshes and rt.detect_mixamo_prefix(arm) == P
    assert int(arm.animation_data.action.frame_range[1]) == end
    bpy.context.scene.frame_set(end)
    bpy.context.view_layer.update()
    assert (arm.matrix_world @ arm.pose.bones[P + "Hips"].head).xy.length > 3.0       # 移動あり

    arm, meshes = _reimport(paths['MOTION'])
    assert not meshes and rt.detect_mixamo_prefix(arm) == P
    for f in (1, end // 2, end):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        assert (arm.matrix_world @ arm.pose.bones[P + "Hips"].head).xy.length < 1e-3  # In Place

    # 条件 6：サンプル動きを一度も開いていない。実体も取得していない
    ui.stop_server()
    assert open(log, encoding="utf-8").read().strip() == ""
    motions = os.path.join(home, "motionbricks", "assets", "motions")
    sizes = [os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(motions) for f in fs
             if f.endswith((".bvh", ".csv"))]
    assert sizes and max(sizes) < 1024, max(sizes)                                   # LFS のポインタのみ
    print(f"ACCEPT ok, exports in {out}", flush=True)
