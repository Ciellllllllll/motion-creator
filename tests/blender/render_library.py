"""ポーズ集の各項目（既定のタイミング）をキャラクターに当てて画像にする（目視確認用）。MB_SAMPLE_DIR へ。"""
import math
import os

import bpy

import common
from motionbricks_blender import library, poses, rig_transfer as rt, soma_rig


def teardown_module():
    common.stop_real()


def test_render():
    out = os.environ["MB_SAMPLE_DIR"]
    os.makedirs(out, exist_ok=True)
    client = common.start_real()
    scene = bpy.context.scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    char = common.load_character()
    src = soma_rig.build_armature(common.skeleton(), "src", (1, 1, 1))
    bpy.ops.mesh.primitive_plane_add(size=30)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = 1.6
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.location = (3.5, -3.5, 0.9)
    cam.rotation_euler = (math.radians(86), 0, math.radians(45))
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.show_shadows = True
    scene.render.resolution_x = scene.render.resolution_y = 300
    scene.render.image_settings.file_format = 'PNG'
    for i, item in enumerate(library.items()):
        window = client.sample(item["file"], item["default_time"], 0, poses.LOCAL_PLACEMENT,
                               static=bool(item.get("static")))
        soma_rig.write_action(src, [window[0], window[0]], "p")
        rt.transfer_action(src, char, "c")
        scene.frame_set(1)
        scene.render.filepath = os.path.join(out, f"lib_{i:02d}.png")
        bpy.ops.render.render(write_still=True)
        print("LIB", i, item["id"], item["name"], flush=True)
