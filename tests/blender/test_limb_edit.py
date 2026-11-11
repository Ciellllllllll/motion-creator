import json

import bpy
from mathutils import Vector

import common
from motionbricks_blender import limb_edit, poses, soma_rig

BODY = ("Hips", "Chest", "Head", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin",
        "RightFoot", "RightToeBase", "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand")


def teardown():
    poses.set_client_factory(None)


def _mb():
    return bpy.context.scene.mb


def _heads(arm):
    bpy.context.view_layer.update()
    return {n: (arm.pose.bones[n].head.copy()) for n in BODY}


def _begin(library_id="walk", index=1):
    common.start_fake()
    mb = _mb()
    mb.index = index
    mb.poses[index].library_id = library_id
    arm = mb.poses[index].armature
    before = _heads(arm)
    assert bpy.ops.mb.limb_edit_begin() == {'FINISHED'}
    return mb, mb.poses[index], arm, before


def _move(arm, handle, delta):
    """ハンドルをアーマチュア座標で *delta* だけ動かし、解き直す。"""
    pb = arm.pose.bones[handle]
    pb.location = pb.location + arm.data.bones[handle].matrix_local.to_3x3().inverted() @ Vector(delta)
    bpy.context.view_layer.update()
    limb_edit.solve_active(bpy.context.scene)
    bpy.context.view_layer.update()


def test_begin_does_not_change_pose():
    mb, entry, arm, before = _begin()
    assert mb.edit_index == 1 and bpy.context.mode == 'POSE'
    for name in limb_edit.HANDLE_NAMES:
        assert name in arm.pose.bones
    limb_edit.solve_active(bpy.context.scene)
    after = _heads(arm)
    assert max((after[n] - before[n]).length for n in BODY) < 1e-3
    # ハンドルは手首・足首の位置にある
    assert (arm.pose.bones["MB_右足"].head - before["RightFoot"]).length < 1e-4
    assert (arm.pose.bones["MB_左手"].head - before["LeftHand"]).length < 1e-4


def test_drag_foot_moves_knee():
    mb, entry, arm, before = _begin()
    foot_q = arm.pose.bones["RightFoot"].matrix.to_quaternion()
    delta = (0.0, -0.10, 0.05)
    _move(arm, "MB_右足", delta)
    after = _heads(arm)
    assert (after["RightFoot"] - (before["RightFoot"] + Vector(delta))).length < 0.01
    assert (after["RightShin"] - before["RightShin"]).length > 0.01           # 膝がついてくる
    assert (after["RightLeg"] - before["RightLeg"]).length < 1e-4             # 股関節は動かない
    for name in ("LeftLeg", "LeftShin", "LeftFoot", "LeftHand", "Head"):
        assert (after[name] - before[name]).length < 1e-4
    turned = foot_q.rotation_difference(arm.pose.bones["RightFoot"].matrix.to_quaternion()).angle
    assert turned < 0.01                                                     # 足の向きは保つ


def test_handler_solves_on_update():
    mb, entry, arm, before = _begin()
    pb = arm.pose.bones["MB_左手"]
    pb.location = pb.location + arm.data.bones["MB_左手"].matrix_local.to_3x3().inverted() @ Vector((0, 0, 0.12))
    for _ in range(3):
        bpy.context.view_layer.update()
    after = _heads(arm)
    assert (after["LeftHand"] - (before["LeftHand"] + Vector((0, 0, 0.12)))).length < 0.01


def test_pole_changes_knee_direction():
    mb, entry, arm, before = _begin("crouch")
    _move(arm, "MB_左ひざ", (0.4, 0.0, 0.0))
    after = _heads(arm)
    assert after["LeftShin"].x - before["LeftShin"].x > 0.02
    assert (after["LeftFoot"] - before["LeftFoot"]).length < 0.01


def test_lower_hips_keeps_feet():
    mb, entry, arm, before = _begin("stand")
    hips = arm.pose.bones["Hips"]
    hips.location = hips.location + arm.data.bones["Hips"].matrix_local.to_3x3().inverted() @ Vector((0, 0, -0.2))
    bpy.context.view_layer.update()
    limb_edit.solve_active(bpy.context.scene)
    after = _heads(arm)
    assert abs((after["Hips"].z - before["Hips"].z) + 0.2) < 1e-3
    for foot in ("LeftFoot", "RightFoot"):
        assert (after[foot] - before[foot]).length < 0.01
    assert (after["LeftShin"] - before["LeftShin"]).length > 0.03            # 膝が曲がる


def test_out_of_reach_is_clamped():
    mb, entry, arm, before = _begin("stand")
    _move(arm, "MB_右手", (0.0, -3.0, 0.0))
    after = _heads(arm)
    reach = (before["RightForeArm"] - before["RightArm"]).length + (before["RightHand"] - before["RightForeArm"]).length
    assert (after["RightHand"] - after["RightArm"]).length <= reach + 1e-4
    assert all(v == v for v in after["RightHand"])                           # NaN にならない


def test_end_stores_edit_and_keeps_velocity():
    mb, entry, arm, before = _begin("walk")
    raw = json.loads(entry.window)
    _move(arm, "MB_左手", (0.0, -0.15, 0.2))
    edited_hand = _heads(arm)["LeftHand"]
    assert bpy.ops.mb.limb_edit_end() == {'FINISHED'}
    assert mb.edit_index == -1 and bpy.context.mode == 'OBJECT' and entry.edited
    assert (_heads(arm)["LeftHand"] - edited_hand).length < 1e-3             # 確定後も同じ姿勢
    window = poses.local_window(entry)
    for k in range(3):
        step_raw = Vector(raw[k + 1][0][:3]) - Vector(raw[k][0][:3])
        step_new = Vector(window[k + 1][0][:3]) - Vector(window[k][0][:3])
        assert (step_raw - step_new).length < 1e-3                           # 進む速さはそのまま
    world = poses.entry_window(entry)
    assert len(world) == 4


def test_edit_survives_resample():
    mb, entry, arm, before = _begin("walk")
    _move(arm, "MB_左手", (0.0, 0.0, 0.3))
    bpy.ops.mb.limb_edit_end()
    raised = _heads(arm)["LeftHand"].z
    entry.time = 1.0 / poses.library_item("walk")["duration"]
    assert entry.edited
    plain = soma_rig.world_matrices(common.skeleton()["parents"], json.loads(entry.window)[0])
    names = common.skeleton()["joint_names"]
    assert _heads(arm)["LeftHand"].z - plain[names.index("LeftHand")].translation.z > 0.1
    assert abs(_heads(arm)["LeftHand"].z - raised) < 0.25


def test_reset():
    mb, entry, arm, before = _begin("walk")
    _move(arm, "MB_右足", (0.0, 0.0, 0.2))
    bpy.ops.mb.limb_edit_end()
    assert bpy.ops.mb.limb_edit_reset() == {'FINISHED'}
    assert entry.edited == ""
    after = _heads(arm)
    assert max((after[n] - before[n]).length for n in BODY) < 1e-3


def test_no_leftovers():
    mb, entry, arm, before = _begin("walk")
    n_objects = len(bpy.data.objects)
    bpy.ops.mb.limb_edit_end()
    assert len(arm.data.bones) == 77
    assert not any(n in arm.data.bones for n in limb_edit.HANDLE_NAMES)
    assert len(bpy.data.objects) <= n_objects
    assert bpy.ops.mb.limb_edit_end.poll() is False


def test_switching_pose_ends_edit():
    mb, entry, arm, before = _begin("walk")
    _move(arm, "MB_左手", (0.0, 0.0, 0.2))
    limb_edit.finish_if_active(bpy.context.scene)
    assert mb.edit_index == -1 and entry.edited and len(arm.data.bones) == 77


def test_handles_are_easy_to_see():
    mb, entry, arm, before = _begin("walk")
    for handle, pole, *_ in limb_edit.LIMBS:
        for name, size in ((handle, 0.05), (pole, 0.03)):
            pb = arm.pose.bones[name]
            assert pb.custom_shape is not None and pb.custom_shape.type == 'MESH'
            assert pb.use_custom_shape_bone_size is False
            assert min(pb.custom_shape_scale_xyz) >= size                 # 画面でつかめる大きさ
    shapes = {arm.pose.bones[n].custom_shape.name for n in limb_edit.HANDLE_NAMES}
    assert len(shapes) == 2                                               # 手足用と向き用で形を変える
    assert all(bpy.data.objects[s].users_collection == () for s in shapes)   # シーンには出さない
