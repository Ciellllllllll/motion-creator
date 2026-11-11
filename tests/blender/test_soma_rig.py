import math

import bpy
from mathutils import Matrix, Vector

import common
from motionbricks_blender import soma_rig


def _arm():
    return soma_rig.build_armature(common.skeleton(), "MB_test", (0.2, 0.8, 0.4))


def test_build_has_all_joints():
    skel = common.skeleton()
    arm = _arm()
    assert [b.name for b in arm.data.bones] and set(b.name for b in arm.data.bones) == set(skel["joint_names"])
    assert arm.data.bones["LeftShin"].parent.name == "LeftLeg"
    heads = common.world_heads(arm)
    assert abs(heads["Hips"].z - skel["rest"][0][2]) < 1e-4


def test_roundtrip_pose():
    skel = common.skeleton()
    arm = _arm()
    pose = common.random_pose(skel, 1)
    soma_rig.apply_pose(arm, pose)
    bpy.context.view_layer.update()
    back = soma_rig.read_pose(arm)
    worst = max(common.quat_angle_deg(a, b) for a, b in zip(pose, back))
    assert worst < 0.01, worst
    assert (Vector(pose[0][:3]) - Vector(back[0][:3])).length < 1e-4


def test_world_positions_match_fk():
    skel = common.skeleton()
    arm = _arm()
    for case in common.fk_cases():
        soma_rig.apply_pose(arm, case["pose"])
        heads = common.world_heads(arm)
        for name, w in zip(skel["joint_names"], case["world"]):
            assert (heads[name] - Vector(w[:3])).length < 1e-3, name


def test_object_transform_does_not_change_world_pose():
    skel = common.skeleton()
    arm = _arm()
    arm.location = (1.0, 2.0, 0.0)
    arm.rotation_euler = (0, 0, 1.1)
    bpy.context.view_layer.update()
    case = common.fk_cases()[-1]
    soma_rig.apply_pose(arm, case["pose"])
    heads = common.world_heads(arm)
    for name, w in zip(skel["joint_names"], case["world"]):
        assert (heads[name] - Vector(w[:3])).length < 1e-3, name
    back = soma_rig.read_pose(arm)
    assert (Vector(back[0][:3]) - Vector(case["pose"][0][:3])).length < 1e-4


def test_action_frames():
    skel = common.skeleton()
    arm = _arm()
    frames = common.walk_frames()
    action = soma_rig.write_action(arm, frames, "MB_walk")
    assert arm.animation_data.action == action
    scene = bpy.context.scene
    scene.frame_set(30)
    back = soma_rig.read_pose(arm)
    for a, b in zip(frames[29], back):
        assert common.quat_angle_deg(a, b) < 0.05
    assert (Vector(back[0][:3]) - Vector(frames[29][0][:3])).length < 1e-3
    assert tuple(int(v) for v in action.frame_range) == (1, len(frames))
