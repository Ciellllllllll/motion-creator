import math

import bpy
from mathutils import Vector

import common
from motionbricks_blender import rig_transfer as rt
from motionbricks_blender import soma_rig

LIMBS = [("LeftLeg", "LeftShin", "LeftUpLeg", "LeftLeg"), ("LeftShin", "LeftFoot", "LeftLeg", "LeftFoot"),
         ("RightLeg", "RightShin", "RightUpLeg", "RightLeg"), ("RightShin", "RightFoot", "RightLeg", "RightFoot"),
         ("LeftArm", "LeftForeArm", "LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand", "LeftForeArm", "LeftHand"),
         ("RightArm", "RightForeArm", "RightArm", "RightForeArm"),
         ("RightForeArm", "RightHand", "RightForeArm", "RightHand"),
         ("Spine1", "Spine2", "Spine", "Spine1"), ("Spine2", "Chest", "Spine1", "Spine2"),
         ("Chest", "Neck1", "Spine2", "Neck")]


def _heads(arm, frame):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    return {pb.name: arm.matrix_world @ pb.head for pb in arm.pose.bones}


def test_limb_directions_match():
    src, _ = common.soma_with_walk()
    char = common.load_character()
    prefix = rt.detect_mixamo_prefix(char)
    action = rt.transfer_action(src, char, "MB_転送")
    assert char.animation_data.action == action
    worst = (0.0, "")
    for frame in (1, 10, 20, 30):
        s, c = _heads(src, frame), _heads(char, frame)
        for a, b, ma, mb in LIMBS:
            ds = (s[b] - s[a]).normalized()
            dc = (c[prefix + mb] - c[prefix + ma]).normalized()
            worst = max(worst, (math.degrees(ds.angle(dc)), f"{a}@{frame}"))
    assert worst[0] < 1.0, worst
    # 背骨は関節の数が違うので、腰→頭の向きは近ければよい
    s, c = _heads(src, 20), _heads(char, 20)
    spine = (s["Head"] - s["Hips"]).angle(c[prefix + "Head"] - c[prefix + "Hips"])
    assert math.degrees(spine) < 8.0, spine


def test_root_travel_scaled_by_height():
    src, _ = common.soma_with_walk()
    char = common.load_character()
    prefix = rt.detect_mixamo_prefix(char)
    rt.transfer_action(src, char, "MB_転送")
    s0, s1 = _heads(src, 1)["Hips"], _heads(src, 30)["Hips"]
    c0, c1 = _heads(char, 1)[prefix + "Hips"], _heads(char, 30)[prefix + "Hips"]
    ratio = rt.mb_to_mixamo(src, char).root_scale
    assert 0.5 < ratio < 1.5
    assert (s1 - s0).xy.length > 0.3
    assert ((c1 - c0).xy - (s1 - s0).xy * ratio).length < 0.01, ((c1 - c0).xy, (s1 - s0).xy, ratio)


def test_roundtrip_error_under_1cm():
    src, _ = common.soma_with_walk()
    char = common.load_character()
    rt.transfer_action(src, char, "MB_転送")
    back = soma_rig.build_armature(common.skeleton(), "MB_back", (1, 0, 0))
    rt.transfer_back(char, back, "MB_戻し")
    skip = ("Thumb", "Index", "Middle", "Ring", "Pinky", "Eye", "Jaw", "End", "Neck2")
    worst = 0.0
    for frame in (1, 12, 25, 34):
        a, b = _heads(src, frame), _heads(back, frame)
        for name in a:
            if not any(k in name for k in skip):
                worst = max(worst, (a[name] - b[name]).length)
    assert worst < 0.01, worst


def test_fingers_stay_at_rest():
    src, _ = common.soma_with_walk()
    char = common.load_character()
    prefix = rt.detect_mixamo_prefix(char)
    rt.transfer_action(src, char, "MB_転送")
    bpy.context.scene.frame_set(15)
    for pb in char.pose.bones:
        if any(k in pb.name for k in ("Thumb", "Index", "Middle", "Ring", "Pinky")):
            assert pb.matrix_basis.to_quaternion().angle < 1e-4, pb.name


def test_prefix_variants():
    for new_prefix in ("mixamorig1:", ""):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        src, _ = common.soma_with_walk()
        char = common.load_character()
        old = rt.detect_mixamo_prefix(char)
        for bone in char.data.bones:
            if bone.name.startswith(old):
                bone.name = new_prefix + bone.name[len(old):]
        assert rt.detect_mixamo_prefix(char) == new_prefix
        tr = rt.mb_to_mixamo(src, char)
        assert len(tr.mapping) >= 22
        rt.transfer_action(src, char, "MB_転送")
        s, c = _heads(src, 20), _heads(char, 20)
        ds = (s["LeftShin"] - s["LeftLeg"]).normalized()
        dc = (c[new_prefix + "LeftLeg"] - c[new_prefix + "LeftUpLeg"]).normalized()
        assert math.degrees(ds.angle(dc)) < 3.0


def test_not_mixamo():
    src, _ = common.soma_with_walk()
    char = common.load_character()
    for i, bone in enumerate(char.data.bones):
        bone.name = f"bone_{i:03d}"
    try:
        rt.transfer_action(src, char, "MB_転送")
        assert False, "エラーになるはず"
    except rt.NotMixamoError as e:
        assert any(ord(ch) > 0x3000 for ch in str(e))


def test_existing_action_kept():
    src, _ = common.soma_with_walk()
    char = common.load_character()
    prefix = rt.detect_mixamo_prefix(char)
    old, _, _ = rt._new_action_for(char, "既存の動き")
    pb = char.pose.bones[prefix + "Head"]
    pb.rotation_mode = 'QUATERNION'
    for f in (1, 10):
        pb.rotation_quaternion = (1, 0.1 * f, 0, 0)
        pb.keyframe_insert("rotation_quaternion", frame=f)
    before = common.action_signature(old)
    new = rt.transfer_action(src, char, "MB_転送")
    assert new != old and "既存の動き" in bpy.data.actions and old.use_fake_user
    assert common.action_signature(old) == before
