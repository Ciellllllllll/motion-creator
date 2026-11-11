"""ゲームエンジンでの確認用に FBX を書き出す（MB_SAMPLE_DIR へ）。前へ 4 m 歩いて止まる動き。"""
import json
import os

import bpy

import common
from motionbricks_blender import poses


def teardown():
    poses.set_client_factory(None)


def teardown_module():
    common.stop_real()


def test_make():
    out = os.environ["MB_SAMPLE_DIR"]
    os.makedirs(out, exist_ok=True)
    common.start_real()
    mb = bpy.context.scene.mb
    mb.poses[0].armature.rotation_euler.z = 0.9          # わざと回して置く（書き出しで正面にそろうはず）
    import math
    mb.poses[1].armature.location = (4.0 * math.sin(0.9), -4.0 * math.cos(0.9), 0.0)   # 最初の正面へ 4 m
    mb.poses[1].armature.rotation_euler.z = 0.9
    mb.character_path = str(common.character_path())
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error
    frames = int(mb.result_action.frame_range[1])
    for name, mode, in_place in (("mb_bake_walk", 'BAKE', False), ("mb_motion_inplace", 'MOTION', True)):
        mb.export_mode, mb.in_place = mode, in_place
        assert bpy.ops.mb.export(filepath=os.path.join(out, name + ".fbx")) == {'FINISHED'}, mb.error
    with open(os.path.join(out, "expected.json"), "w") as f:
        json.dump({"frames": frames, "seconds": (frames - 1) / 30.0}, f)
    print("SAMPLES", out, frames, flush=True)
