import math
import os
import tempfile

import bpy
from mathutils import Vector

import common
from motionbricks_blender import poses, soma_rig


def teardown():
    poses.set_client_factory(None)


def _mb():
    return bpy.context.scene.mb


def _hips_world(entry):
    bpy.context.view_layer.update()
    return entry.armature.matrix_world @ entry.armature.pose.bones["Hips"].head


def test_initial_layout():
    common.start_fake()
    mb = _mb()
    assert [e.kind for e in mb.poses] == ['START', 'END']
    assert [poses.entry_label(mb, i) for i in range(2)] == ["最初", "最後"]
    a, b = mb.poses[0].armature, mb.poses[1].armature
    assert a.location.xy.length < 1e-6 and (b.location.xy - Vector((0, -2))).length < 1e-6
    assert all(e.library_id == "stand" for e in mb.poses)
    assert abs(_hips_world(mb.poses[1]).y + 2.0) < 0.15 and _hips_world(mb.poses[0]).z > 0.8
    # 人形は床の上だけを動く
    assert tuple(a.lock_location) == (False, False, True) and tuple(a.lock_rotation) == (True, True, False)


def test_start_twice_keeps_poses():
    common.start_fake()
    _mb().poses[1].armature.location.x = 1.0
    common.start_fake()
    assert len(_mb().poses) == 2 and abs(_mb().poses[1].armature.location.x - 1.0) < 1e-6


def test_add_remove_move():
    common.start_fake()
    mb = _mb()
    mb.index = 0
    assert bpy.ops.mb.pose_add() == {'FINISHED'}
    assert bpy.ops.mb.pose_add() == {'FINISHED'}
    assert [e.kind for e in mb.poses] == ['START', 'MIDDLE', 'MIDDLE', 'END']
    assert [poses.entry_label(mb, i) for i in range(4)] == ["最初", "途中 1", "途中 2", "最後"]
    first_mid = mb.poses[1].armature
    mb.index = 0
    assert bpy.ops.mb.pose_remove.poll() is False
    assert bpy.ops.mb.pose_move.poll() is False
    mb.index = 3
    assert bpy.ops.mb.pose_remove.poll() is False
    mb.index = 1
    assert bpy.ops.mb.pose_move(direction='UP') == {'CANCELLED'}       # 最初より前へは動かない
    assert bpy.ops.mb.pose_move(direction='DOWN') == {'FINISHED'}
    assert mb.poses[2].armature == first_mid and mb.index == 2
    assert bpy.ops.mb.pose_move(direction='DOWN') == {'CANCELLED'}     # 最後より後ろへは動かない
    n_objects = len(bpy.data.objects)
    assert bpy.ops.mb.pose_remove() == {'FINISHED'}
    assert [e.kind for e in mb.poses] == ['START', 'MIDDLE', 'END'] and len(bpy.data.objects) == n_objects - 1


def test_add_goes_between_neighbours():
    common.start_fake()
    mb = _mb()
    mb.index = 0
    bpy.ops.mb.pose_add()
    assert (mb.poses[1].armature.location.xy - Vector((0, -1))).length < 1e-6
    mb.index = 2                      # 最後を選んで追加しても、最後の手前に入る
    bpy.ops.mb.pose_add()
    assert [e.kind for e in mb.poses] == ['START', 'MIDDLE', 'MIDDLE', 'END']
    assert (mb.poses[2].armature.location.xy - Vector((0, -1.5))).length < 1e-6


def test_time_slider_resamples():
    fake = common.start_fake()
    entry = _mb().poses[1]
    entry.library_id = "walk"
    before = soma_rig.read_pose(entry.armature)
    n = len(fake.calls)
    entry.time = 1.0 / poses.library_item("walk")["duration"]
    assert len(fake.calls) == n + 1
    call = fake.calls[-1]
    assert call["clip"] == "walk.bvh" and abs(call["time"] - 1.0) < 1e-3 and call["anchor"] == 0
    bpy.context.view_layer.update()
    after = soma_rig.read_pose(entry.armature)
    assert max(common.quat_angle_deg(a, b) for a, b in zip(before, after)) > 1.0


def test_start_uses_anchor_3_and_base_is_static():
    fake = common.start_fake()
    mb = _mb()
    mb.poses[0].library_id = "walk"
    assert fake.calls[-1]["anchor"] == 3
    mb.poses[0].library_id = "base"
    assert fake.calls[-1]["static"] is True


def test_drag_moves_window():
    common.start_fake()
    entry = _mb().poses[1]
    entry.library_id = "walk"
    arm = entry.armature
    arm.location = (1.0, 2.0, 0.0)
    arm.rotation_euler = (0, 0, math.pi / 2)
    bpy.context.view_layer.update()
    window = poses.entry_window(entry)
    assert len(window) == 4 and len(window[0]) == 77
    hips = _hips_world(entry)
    assert (Vector(window[0][0][:3]) - hips).length < 1e-3           # 表示と同じ場所
    assert (Vector(window[0][0][:2]) - Vector((1.0, 2.0))).length < 0.05
    # 人形は自分の -Y を向く。90 度回すと +X を向いて進む
    travel = Vector(window[3][0][:2]) - Vector(window[0][0][:2])
    assert travel.x > 0.03 and abs(travel.y) < travel.x


def test_persist_after_reload():
    common.start_fake()
    mb = _mb()
    mb.index = 0
    bpy.ops.mb.pose_add()
    mb.poses[1].library_id = "walk"
    mb.poses[1].armature.location = (0.5, -1.0, 0.0)
    mb.status = 'RUNNING'
    bpy.context.view_layer.update()
    window = poses.entry_window(mb.poses[1])
    path = os.path.join(tempfile.mkdtemp(), "保存テスト.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    mb = _mb()
    assert mb.status == 'STOPPED'
    assert [e.kind for e in mb.poses] == ['START', 'MIDDLE', 'END'] and mb.poses[1].library_id == "walk"
    assert mb.poses[1].armature is not None
    again = poses.entry_window(mb.poses[1])
    assert max(abs(a - b) for pa, pb in zip(window, again) for ja, jb in zip(pa, pb) for a, b in zip(ja, jb)) < 1e-4
    # 起動し直しても並べたポーズはそのまま
    common.start_fake()
    assert len(_mb().poses) == 3


def test_add_after_user_deleted_a_puppet():
    common.start_fake()
    mb = _mb()
    bpy.data.objects.remove(mb.poses[1].armature, do_unlink=True)
    mb.index = 0
    assert bpy.ops.mb.pose_add() == {'FINISHED'}
    assert len(mb.poses) == 3 and all(e.armature is not None for e in mb.poses)
