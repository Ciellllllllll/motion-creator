"""FBX の動きを、等間隔のフレームの画像にして並べる（目視確認用）。

blender -b --factory-startup --python tools/render_strip.py -- <fbx> <出力フォルダ> [枚数] [front|side]
"""
import math
import os
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
fbx, out = args[0], args[1]
count = int(args[2]) if len(args) > 2 else 12
view = args[3] if len(args) > 3 else "side"
os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = 30
bpy.ops.import_scene.fbx(filepath=fbx)
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
hips = next(pb for pb in arm.pose.bones if pb.name.endswith("Hips"))
end = int(arm.animation_data.action.frame_range[1])
bpy.ops.mesh.primitive_plane_add(size=40)
cam_data = bpy.data.cameras.new("cam")
cam_data.type = 'ORTHO'
cam_data.ortho_scale = 2.4
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
light = bpy.data.objects.new("sun", bpy.data.lights.new("sun", 'SUN'))
scene.collection.objects.link(light)
light.rotation_euler = (0.6, 0.2, 0.5)
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'STUDIO'
scene.display.shading.show_shadows = True
scene.render.resolution_x = scene.render.resolution_y = 360
scene.render.image_settings.file_format = 'PNG'
for i in range(count):
    f = 1 + round(i * (end - 1) / max(1, count - 1))
    scene.frame_set(f)
    bpy.context.view_layer.update()
    p = arm.matrix_world @ hips.head
    if view == "front":
        cam.location = (p.x, p.y - 6, 1.0)
        cam.rotation_euler = (math.pi / 2, 0, 0)
    else:
        cam.location = (p.x + 6, p.y, 1.0)
        cam.rotation_euler = (math.pi / 2, 0, math.pi / 2)
    scene.render.filepath = os.path.join(out, f"{view}_{i:02d}.png")
    bpy.ops.render.render(write_still=True)
print("RENDERED", count, "frames to", out, "end", end)
