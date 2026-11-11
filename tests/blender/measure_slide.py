"""足滑りの測定（判断材料。合否は付けない）。"""
import json

import bpy

import common
import test_generate as tg
from motionbricks_blender import generate, poses


def teardown():
    poses.set_client_factory(None)


def teardown_module():
    common.stop_real()


def _measure(label, arm, names):
    scene = bpy.context.scene
    act = scene.mb.result_action
    contacts = json.loads(act["mb_contacts"])
    cols = {0: names[0], 1: names[1], 2: names[2], 3: names[3]}
    prev, speeds, low = None, [], []
    for f in range(1, len(contacts) + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        pos = {c: (arm.matrix_world @ arm.pose.bones[n].head).copy() for c, n in cols.items()}
        if prev is not None:
            for c in cols:
                if contacts[f - 1][c] > 0.5 and contacts[f - 2][c] > 0.5:
                    speeds.append((pos[c] - prev[c]).xy.length * 30 * 100)
        low.append(min(p.z for p in pos.values()) * 100)
        prev = pos
    speeds.sort()
    mean = sum(speeds) / max(1, len(speeds))
    print(f"SLIDE {label}: contact samples {len(speeds)}, mean {mean:.2f} cm/s, "
          f"p90 {speeds[int(len(speeds) * 0.9)] if speeds else 0:.2f} cm/s, max {speeds[-1] if speeds else 0:.2f} cm/s, "
          f"lowest joint z {min(low):.1f}..{max(low):.1f} cm", flush=True)


def test_measure():
    common.start_real()
    mb = tg._four_poses()
    mb.character_path = str(common.character_path())
    bpy.ops.mb.generate()
    scene = bpy.context.scene
    _measure("SOMA 4poses", generate.result_armature(scene), ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"])
    char = generate.character_armature(scene)
    p = "mixamorig:"
    _measure("Mixamo 4poses", char, [p + "LeftFoot", p + "LeftToeBase", p + "RightFoot", p + "RightToeBase"])


def test_measure_walk_far():
    common.start_real()
    mb = bpy.context.scene.mb
    mb.poses[1].armature.location = (0.0, -6.0, 0.0)
    mb.character_path = str(common.character_path())
    bpy.ops.mb.generate()
    scene = bpy.context.scene
    _measure("SOMA far", generate.result_armature(scene), ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"])
    p = "mixamorig:"
    _measure("Mixamo far", generate.character_armature(scene),
             [p + "LeftFoot", p + "LeftToeBase", p + "RightFoot", p + "RightToeBase"])
