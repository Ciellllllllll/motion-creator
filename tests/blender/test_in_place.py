"""生成結果とキャラクターの In Place 再生を実際の Blender で確認する。"""
import pathlib
import tempfile

import bpy

import common
from motionbricks_blender import export, generate, poses, rig_transfer as rt


def teardown_module():
    common.stop_real()


def _generate(character=False):
    common.start_real()
    mb = bpy.context.scene.mb
    mb.poses[-1].armature.location = (0.6, -3.0, 0)
    if character:
        mb.character_path = str(common.character_path())
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error
    return mb, generate.result_armature(bpy.context.scene)


def _hips(arm):
    scene = bpy.context.scene
    action = arm.animation_data.action
    name = (rt.detect_mixamo_prefix(arm) or '') + 'Hips'
    points = []
    for frame in range(int(action.frame_range[0]), int(action.frame_range[1]) + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        points.append((arm.matrix_world @ arm.pose.bones[name].head).copy())
    return points


def _assert_stationary(arm, original=None):
    points = _hips(arm)
    assert max((p.xy - points[0].xy).length for p in points) < 1e-4
    if original is not None:
        assert max(abs(p.z - q.z) for p, q in zip(points, original)) < 1e-4


def test_enabled_before_generating():
    bpy.context.scene.mb.in_place = True
    mb, source = _generate(character=True)
    _assert_stationary(source)
    _assert_stationary(generate.character_armature(bpy.context.scene))
    assert bpy.ops.mb.regenerate() == {'FINISHED'}, mb.error
    _assert_stationary(source)
    _assert_stationary(generate.character_armature(bpy.context.scene))


def test_toggle_restores_source_and_character():
    mb, source = _generate(character=True)
    char = generate.character_armature(bpy.context.scene)
    actions = [source.animation_data.action, char.animation_data.action]
    signatures = [common.action_signature(a) for a in actions]
    originals = [_hips(source), _hips(char)]
    count = len(bpy.data.actions)
    for _ in range(2):
        mb.in_place = True
        for arm, original in zip((source, char), originals):
            _assert_stationary(arm, original)
        for action, signature in zip(actions, signatures):
            assert [c for c in common.action_signature(action) if not c[0].endswith('.location')] == [
                c for c in signature if not c[0].endswith('.location')]
        mb.in_place = False
        assert [common.action_signature(a) for a in actions] == signatures
    assert len(bpy.data.actions) == count


def test_character_added_after_in_place_generation():
    bpy.context.scene.mb.in_place = True
    mb, source = _generate()
    _assert_stationary(source)
    mb.character_path = str(common.character_path())
    char, action = export.ensure_character_motion(bpy.context.scene)
    _assert_stationary(char)
    assert rt.is_in_place(source)
    mb.in_place = False
    assert not rt.is_in_place(source) and not rt.is_in_place(char)
    source_points, char_points = _hips(source), _hips(char)
    scale = rt.mb_to_mixamo(source, char).root_scale
    assert (source_points[-1].xy - source_points[0].xy).length > 2
    assert max((c.xy - s.xy * scale).length for s, c in zip(source_points, char_points)) < 0.03


def test_toggle_after_save_and_reload():
    mb, source = _generate(character=True)
    char = generate.character_armature(bpy.context.scene)
    names = [source.name, char.name]
    signatures = [common.action_signature(arm.animation_data.action) for arm in (source, char)]
    mb.in_place = True
    # 保存後も元のキーを復元できることを確認する。
    path = pathlib.Path(tempfile.mkdtemp()) / 'in_place.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    bpy.ops.wm.open_mainfile(filepath=str(path))
    assert bpy.context.scene.mb.in_place
    for name in names:
        _assert_stationary(bpy.data.objects[name])
    bpy.context.scene.mb.in_place = False
    assert [common.action_signature(bpy.data.objects[name].animation_data.action) for name in names] == signatures
