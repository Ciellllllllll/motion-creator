"""3D ビューの目印：ポーズ人形の足元に色つきの輪と向きの矢印、頭上に名前を描く。

色は一覧と同じ（最初＝緑、途中＝オレンジ、最後＝青）。Blender はオブジェクトモードで
ボーンの色を表示しないので、どのモードでも見分けられるように自前で描く。
"""

import math

import bpy

from . import poses

RADIUS = 0.33
SEGMENTS = 40
LABEL_HEIGHT = 1.95
_handles = []


def marks(scene) -> list:
    """描くもの（表示中の人形ごとに 1 つ）。画面なしでも計算できる。"""
    mb = getattr(scene, "mb", None)
    out = []
    if mb is None:
        return out
    for i, entry in enumerate(mb.poses):
        arm = entry.armature
        if arm is None:
            continue
        try:
            if arm.hide_get() or arm.hide_viewport:
                continue
        except RuntimeError:        # このビューレイヤーに無い
            continue
        x, y, rot = arm.location.x, arm.location.y, arm.rotation_euler.z
        facing = (math.sin(rot), -math.cos(rot))          # 人形のローカル -Y
        selected = i == mb.index
        radius = RADIUS * (1.12 if selected else 1.0)
        ring = [(x + radius * math.cos(2 * math.pi * k / SEGMENTS), y + radius * math.sin(2 * math.pi * k / SEGMENTS),
                 0.004) for k in range(SEGMENTS)]
        side = (-facing[1], facing[0])
        tip = (x + facing[0] * (radius + 0.22), y + facing[1] * (radius + 0.22), 0.004)
        base = (x + facing[0] * (radius + 0.02), y + facing[1] * (radius + 0.02))
        arrow = [(base[0] + side[0] * 0.09, base[1] + side[1] * 0.09, 0.004), tip,
                 (base[0] - side[0] * 0.09, base[1] - side[1] * 0.09, 0.004)]
        out.append({"label": poses.entry_label(mb, i), "color": poses.COLORS[entry.kind], "center": (x, y, 0.0),
                    "facing": facing, "ring": ring, "arrow": arrow, "selected": selected,
                    "text_at": (x, y, LABEL_HEIGHT)})
    return out


def _draw_view():
    import gpu
    from gpu_extras.batch import batch_for_shader

    items = marks(bpy.context.scene)
    if not items:
        return
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    region = bpy.context.region
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    shader.uniform_float("viewportSize", (region.width, region.height))
    for item in items:
        shader.uniform_float("lineWidth", 4.5 if item["selected"] else 2.5)
        shader.uniform_float("color", (*item["color"], 1.0))
        ring = item["ring"] + [item["ring"][0]]
        batch_for_shader(shader, 'LINE_STRIP', {"pos": ring}).draw(shader)
        batch_for_shader(shader, 'LINE_STRIP', {"pos": item["arrow"]}).draw(shader)
    fill = gpu.shader.from_builtin('UNIFORM_COLOR')
    for item in items:
        fill.uniform_float("color", (*item["color"], 0.16 if item["selected"] else 0.07))
        center = (item["center"][0], item["center"][1], 0.003)
        batch_for_shader(fill, 'TRI_FAN', {"pos": [center] + item["ring"] + [item["ring"][0]]}).draw(fill)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


def _draw_text():
    import blf
    from bpy_extras.view3d_utils import location_3d_to_region_2d

    context = bpy.context
    region, rv3d = context.region, context.region_data
    if region is None or rv3d is None:
        return
    scale = context.preferences.system.ui_scale or 1.0
    for item in marks(context.scene):
        pos = location_3d_to_region_2d(region, rv3d, item["text_at"])
        if pos is None:
            continue
        size = (15 if item["selected"] else 13) * scale
        blf.size(0, size)
        width, _ = blf.dimensions(0, item["label"])
        blf.enable(0, blf.SHADOW)
        blf.shadow(0, 3, 0.0, 0.0, 0.0, 0.9)
        blf.shadow_offset(0, 1, -1)
        blf.color(0, *item["color"], 1.0)
        blf.position(0, pos.x - width / 2, pos.y, 0)
        blf.draw(0, item["label"])
        blf.disable(0, blf.SHADOW)


def register():
    if bpy.app.background:
        return
    space = bpy.types.SpaceView3D
    _handles.append(space.draw_handler_add(_draw_view, (), 'WINDOW', 'POST_VIEW'))
    _handles.append(space.draw_handler_add(_draw_text, (), 'WINDOW', 'POST_PIXEL'))


def unregister():
    while _handles:
        bpy.types.SpaceView3D.draw_handler_remove(_handles.pop(), 'WINDOW')
