"""自作 NF4 で再生成したポーズ集を、アドオンで通しで確認する（仕様 8.3 後半）。

リポジトリの一時コピー（library/ を新ポーズ集に差し替えたもの）で実行する:
  blender -b --factory-startup --python <コピー>/tests/blender/run.py -- blender_check_library
環境変数 MB_NF4_OUT に、描画画像・FBX・結果 JSON を書く。

各クリップについて:
1. クリップを6時点でキャラクターへ当てて描画（目視確認用の画像）
2. ポーズ選択（同じクリップの2ポーズ）→ MotionBricks 生成 → Mixamo 転送 → FBX 書き出し → 再読み込み
"""
import json
import math
import os

import bpy

import common
from motionbricks_blender import generate, library, poses, rig_transfer as rt, soma_rig

P = "mixamorig:"


def teardown_module():
    common.stop_real()


def _camera(scene, ortho=1.9):
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = ortho
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.location = (3.5, -3.5, 0.95)
    cam.rotation_euler = (math.radians(86), 0, math.radians(45))
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.show_shadows = True
    scene.render.resolution_x = scene.render.resolution_y = 260
    scene.render.image_settings.file_format = 'PNG'


def _clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)


def render_clip_frames(client, item, out):
    scene = bpy.context.scene
    _clear()
    char = common.load_character()
    src = soma_rig.build_armature(common.skeleton(), "src", (1, 1, 1))
    bpy.ops.mesh.primitive_plane_add(size=30)
    _camera(scene)
    files = []
    for k in range(6):
        t = min(item["duration"] * k / 5, item["duration"] - 1 / 30)
        window = client.sample(item["file"], t, 0, poses.LOCAL_PLACEMENT, static=False)
        assert len(window) == 4 and len(window[0]) == 77
        assert all(math.isfinite(v) for j in window[0] for v in j)
        soma_rig.write_action(src, [window[0], window[0]], f"p{k}")
        rt.transfer_action(src, char, f"c{k}")
        scene.frame_set(1)
        path = os.path.join(out, f"{item['id']}_t{k}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        files.append(path)
    return files


def pipeline(item, out):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 30
    common.start_real(scene)
    mb = scene.mb
    mb.index = 0
    bpy.ops.mb.pose_add()
    bpy.ops.mb.pose_add()
    mb.poses[1].library_id = item["id"]
    mb.poses[1].armature.location = (0.0, -1.5, 0.0)
    mb.poses[2].library_id = item["id"]
    mb.poses[2].armature.location = (0.0, -3.0, 0.0)
    mb.character_path = str(common.character_path())
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error
    result = generate.result_armature(scene)
    assert result is not None
    end = int(mb.result_action.frame_range[1])
    fbx = os.path.join(out, f"{item['id']}_BAKE.fbx")
    mb.export_mode, mb.in_place = 'BAKE', False
    assert bpy.ops.mb.export(filepath=fbx) == {'FINISHED'}, mb.error
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.fbx(filepath=fbx)
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    assert meshes and rt.detect_mixamo_prefix(arm) == P
    re_end = int(arm.animation_data.action.frame_range[1])
    assert re_end == end, (re_end, end)
    finite = True
    for f in (1, end // 2, end):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        h = arm.matrix_world @ arm.pose.bones[P + "Hips"].head
        finite &= all(math.isfinite(v) for v in h)
    assert finite
    return {"generated_frames": end, "reimported_frames": re_end, "fbx": fbx, "fbx_bytes": os.path.getsize(fbx),
            "mixamo_prefix": True, "mesh_count": len(meshes)}


def test_nf4_library():
    out = os.environ["MB_NF4_OUT"]
    os.makedirs(out, exist_ok=True)
    items = [i for i in library.items() if not i.get("static")]
    assert len(items) == 10
    enc = {i["text_encoder"]["model_safetensors_sha256"] for i in items}
    assert len(enc) == 1, "ポーズ集に自作エンコーダーの記録がない/混在している"
    results = {"library_dir": str(library.LIBRARY_DIR), "text_encoder_sha256": enc.pop(), "clips": {}}
    client = common.start_real()
    for item in items:
        results["clips"][item["id"]] = {"name": item["name"], "frames_rendered": render_clip_frames(client, item, out)}
        print("RENDERED", item["id"], flush=True)
    for item in items:
        results["clips"][item["id"]]["pipeline"] = pipeline(item, out)
        print("PIPELINE", item["id"], results["clips"][item["id"]]["pipeline"]["generated_frames"], flush=True)
    with open(os.path.join(out, "blender_check.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
