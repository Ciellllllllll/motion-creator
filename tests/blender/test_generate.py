import json
import os

import bpy
from mathutils import Vector

import common
from motionbricks_blender import generate, limb_edit, poses, rig_transfer as rt


def teardown():
    poses.set_client_factory(None)


def teardown_module():
    common.stop_real()


def _mb():
    return bpy.context.scene.mb


def _result_hips(frame):
    arm = generate.result_armature(bpy.context.scene)
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    return arm.matrix_world @ arm.pose.bones["Hips"].head


def _four_poses():
    """最初・途中 2・最後。途中 1 は歩き、途中 2 はしゃがみ（手足も編集）。"""
    mb = _mb()
    mb.index = 0
    bpy.ops.mb.pose_add()
    bpy.ops.mb.pose_add()
    a, b = mb.poses[1], mb.poses[2]
    a.library_id = "walk"
    a.armature.location = (0.0, -1.2, 0.0)
    b.library_id = "crouch"
    b.armature.location = (0.8, -2.4, 0.0)
    b.armature.rotation_euler.z = 0.6
    mb.poses[3].armature.location = (1.5, -3.6, 0.0)
    mb.index = 2
    bpy.ops.mb.limb_edit_begin()
    pb = b.armature.pose.bones["MB_左手"]
    pb.location = pb.location + b.armature.data.bones["MB_左手"].matrix_local.to_3x3().inverted() @ Vector((0, 0, 0.3))
    bpy.context.view_layer.update()
    return mb


def test_generate_end_to_end():
    common.start_real()
    mb = _four_poses()
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error
    assert mb.edit_index == -1 and mb.poses[2].edited          # 編集は自動で確定される
    action = mb.result_action
    assert action is not None and action.name.startswith("MB_生成")
    frames = json.loads(action["mb_pose_frames"])
    assert len(frames) == 3 and frames == sorted(frames)
    scene = bpy.context.scene
    assert scene.frame_start == 1 and scene.frame_end == int(action.frame_range[1]) and scene.render.fps == 30
    for entry, frame in zip(list(mb.poses)[1:], frames):
        want = Vector(poses.entry_window(entry)[0][0][:2])
        got = _result_hips(frame).xy
        assert (got - want).length < 0.10, (entry.kind, frame, got, want)
    start = Vector(poses.entry_window(mb.poses[0])[3][0][:2])
    assert (_result_hips(1).xy - start).length < 0.05
    # 編集で上げた左手が結果にも出ている
    arm = generate.result_armature(scene)
    scene.frame_set(frames[1])
    bpy.context.view_layer.update()
    hand = (arm.matrix_world @ arm.pose.bones["LeftHand"].head).z
    hips = (arm.matrix_world @ arm.pose.bones["Hips"].head).z
    assert hand > hips
    # 生成後はポーズ人形を隠す
    assert all(e.armature.hide_get() for e in mb.poses)


def test_generate_with_character():
    common.start_real()
    mb = _mb()
    path = str(common.character_path())
    stamp = (os.path.getmtime(path), os.path.getsize(path))
    mb.character_path = path
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error
    char = generate.character_armature(bpy.context.scene)
    assert char is not None and rt.detect_mixamo_prefix(char) == "mixamorig:"
    act = char.animation_data.action
    assert act.name.startswith("MB_Mixamo") and int(act.frame_range[1]) == int(mb.result_action.frame_range[1])
    assert (os.path.getmtime(path), os.path.getsize(path)) == stamp
    n = len(bpy.data.objects)
    bpy.ops.mb.generate()                                       # 2 回目はキャラクターを読み直さない
    assert len(bpy.data.objects) == n and generate.character_armature(bpy.context.scene) == char


def test_far_poses_get_intermediate_targets():
    common.start_real()
    mb = _mb()
    mb.poses[1].armature.location = (0.0, -7.0, 0.0)
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error
    end_frame = json.loads(mb.result_action["mb_pose_frames"])[-1]
    assert (_result_hips(end_frame).xy - Vector((0.0, -7.0))).length < 0.10
    assert end_frame > 90                                       # 1 回の生成（最長 58 フレーム）では届かない距離


def test_generate_from_base_pose():
    common.start_real()
    mb = _mb()
    for i in (0, 1):
        mb.poses[i].library_id = "base"
    mb.index = 1
    bpy.ops.mb.limb_edit_begin()
    arm = mb.poses[1].armature
    pb = arm.pose.bones["MB_右手"]
    pb.location = pb.location + arm.data.bones["MB_右手"].matrix_local.to_3x3().inverted() @ Vector((0, -0.2, 0.4))
    bpy.context.view_layer.update()
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error
    end = json.loads(mb.result_action["mb_pose_frames"])[-1]
    assert (_result_hips(end).xy - Vector((0.0, -2.0))).length < 0.10


def test_regenerate_differs():
    common.start_real()
    mb = _mb()
    bpy.ops.mb.generate()
    first = mb.result_action
    n1 = int(first.frame_range[1])
    sig1 = common.action_signature(first)
    assert bpy.ops.mb.regenerate() == {'FINISHED'}
    second = mb.result_action
    assert second != first and mb.variant >= 1
    assert int(second.frame_range[1]) != n1 or common.action_signature(second) != sig1
    assert common.action_signature(first) == sig1               # 前の結果は残る
    bpy.ops.mb.generate()
    assert mb.variant == 0


def test_play_toggle_poll():
    common.start_real()
    assert bpy.ops.mb.play.poll() is False                      # 結果がまだ無い
    assert bpy.ops.mb.regenerate.poll() is False
    bpy.ops.mb.generate()
    assert bpy.ops.mb.play.poll() is True and bpy.ops.mb.regenerate.poll() is True


def test_server_dies_midway():
    common.start_real()
    mb = _mb()
    bpy.ops.mb.generate()
    kept = mb.result_action
    sig = common.action_signature(kept)
    common.kill_real()
    try:
        result = bpy.ops.mb.generate()
    except RuntimeError:        # スクリプトからの呼び出しでは、エラー報告が例外になる
        result = {'CANCELLED'}
    assert result == {'CANCELLED'}
    assert mb.status == 'ERROR' and any(ord(ch) > 0x3000 for ch in mb.error)
    assert mb.result_action == kept and common.action_signature(kept) == sig


def _ball_slide(char, intervals):
    """接地区間での足（母指球）の水平移動の平均速度 cm/s と、最も低い足裏の高さ cm。"""
    from motionbricks_blender import foot_lock
    feet = foot_lock.Feet(char)
    scene = bpy.context.scene
    speeds, lows = [], []
    for side in ("L", "R"):
        for s, e in intervals[side]:
            prev = None
            for f in range(s, e + 1):
                scene.frame_set(f)
                bpy.context.view_layer.update()
                pts = feet.points(side, {pb.name: pb.matrix.copy() for pb in char.pose.bones})
                lows.append(min(p.z for p in pts.values()) * 100)
                if prev is not None:
                    speeds.append((pts["ball"] - prev).xy.length * 30 * 100)
                prev = pts["ball"].copy()
    return sum(speeds) / max(1, len(speeds)), min(lows), max(lows)


def test_feet_locked_on_character():
    from motionbricks_blender import foot_lock
    common.start_real()
    mb = _mb()
    mb.poses[1].armature.location = (0.0, -6.0, 0.0)
    mb.character_path = str(common.character_path())
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error
    scene = bpy.context.scene
    char = generate.character_armature(scene)
    intervals = foot_lock.contact_intervals(json.loads(mb.result_action["mb_contacts"]), start=1)
    assert len(intervals["L"]) >= 2 and len(intervals["R"]) >= 2
    mean, low_min, low_max = _ball_slide(char, intervals)
    print("LOCKED", round(mean, 2), round(low_min, 2), round(low_max, 2), flush=True)
    assert mean < 3.0, mean                       # 設計の基準：平均 3 cm/s 以内
    assert low_min > -0.5 and low_max < 1.5, (low_min, low_max)   # 接地中の足裏は床の上
    # 腰の動きは変えない（SOMA の結果と同じ道筋）
    arm = generate.result_armature(scene)
    ratio = rt.mb_to_mixamo(arm, char).root_scale
    end = int(mb.result_action.frame_range[1])
    scene.frame_set(end)
    bpy.context.view_layer.update()
    c = (char.matrix_world @ char.pose.bones["mixamorig:Hips"].head).xy
    s = (arm.matrix_world @ arm.pose.bones["Hips"].head).xy
    assert (c - s * ratio).length < 0.03


def test_contact_intervals():
    from motionbricks_blender import foot_lock
    rows = [[0.9, 0.1, 0, 0]] * 5 + [[0.1, 0.1, 0, 0.8]] * 2 + [[0.2, 0.9, 0, 0]] * 4 + [[0, 0, 0.9, 0.9]] * 1
    got = foot_lock.contact_intervals(rows, start=1)
    assert got["L"] == [[1, 11]]                  # 2 フレームの切れ目はつなぐ
    assert got["R"] == []                         # 3 フレーム未満の接地は捨てる


def test_survives_user_deleting_objects():
    """人形や結果の骨格を 3D ビューで消されても、次の生成で作り直して続けられる。"""
    common.start_real()
    mb = _mb()
    scene = bpy.context.scene
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error
    # 1) アウトライナーから外しただけ（データは残る） 2) 完全に削除
    end_arm = mb.poses[1].armature
    for coll in list(end_arm.users_collection):
        coll.objects.unlink(end_arm)
    bpy.data.objects.remove(mb.poses[0].armature, do_unlink=True)
    result = generate.result_armature(scene)
    for coll in list(result.users_collection):
        coll.objects.unlink(result)
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error
    assert all(e.armature is not None and e.armature.name in scene.objects for e in mb.poses)
    assert generate.result_armature(scene).name in scene.objects
    assert (Vector(poses.entry_window(mb.poses[1])[0][0][:2]) - Vector((0, -2))).length < 0.05   # 置いた場所は保つ
    mb.index = 0
    assert bpy.ops.mb.limb_edit_begin() == {'FINISHED'}
    assert bpy.ops.mb.limb_edit_end() == {'FINISHED'}
