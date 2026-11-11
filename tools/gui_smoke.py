"""実際の画面つき Blender で、起動 → ポーズ → 生成までを自動で動かして画面を保存する。

blender --factory-startup --python tools/gui_smoke.py -- <出力フォルダ> [キャラクター FBX]
"""
import os
import sys
import time

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
OUT = args[0]
CHAR = args[1] if len(args) > 1 else ""
os.makedirs(OUT, exist_ok=True)
MODULE = "bl_ext.user_default.motionbricks_blender"
state = {"step": 0, "t0": time.time(), "log": []}


def log(*a):
    line = " ".join(str(x) for x in a)
    state["log"].append(line)
    with open(os.path.join(OUT, "smoke.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ctx():
    window = bpy.context.window_manager.windows[0]
    area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return {"window": window, "screen": window.screen, "area": area, "region": region, "scene": window.scene}


def shot(name):
    c = ctx()
    with bpy.context.temp_override(**c):
        bpy.ops.screen.screenshot(filepath=os.path.join(OUT, name + ".png"))
    log("shot", name)


def tick():
    try:
        return step()
    except Exception as e:
        import traceback
        log("EXCEPTION", traceback.format_exc())
        bpy.ops.wm.quit_blender()
        return None


def step():
    c = ctx()
    mb = c["scene"].mb if hasattr(c["scene"], "mb") else None
    if time.time() - state["t0"] > 420:
        log("TIMEOUT at step", state["step"], mb.status if mb else None, mb.error if mb else None)
        bpy.ops.wm.quit_blender()
        return None
    s = state["step"]
    if s == 0:
        bpy.ops.preferences.addon_enable(module=MODULE)
        space = c["area"].spaces[0]
        space.show_region_ui = True
        ui = next(r for r in c["area"].regions if r.type == 'UI')
        try:
            ui.active_panel_category = "MotionBricks"
        except Exception as e:
            log("category", e)
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o)
        state["step"] = 1
        return 1.0
    if s == 1:
        ui = next(r for r in c["area"].regions if r.type == 'UI')
        try:
            ui.active_panel_category = "MotionBricks"
        except Exception as e:
            log("category2", e)
        shot("1_stopped")
        with bpy.context.temp_override(**c):
            log("start ->", bpy.ops.mb.start())
        state["step"] = 2
        state["shot_progress"] = False
        return 0.3
    if s == 2:
        if mb.status in {'INSTALLING', 'STARTING'}:
            if not state["shot_progress"]:
                shot("2_starting")
                state["shot_progress"] = True
            return 0.3
        log("status", mb.status, mb.error, "after %.1fs" % (time.time() - state["t0"]))
        if mb.status != 'RUNNING':
            shot("error")
            bpy.ops.wm.quit_blender()
            return None
        state["step"] = 3
        return 0.5
    if s == 3:
        with bpy.context.temp_override(**c):
            mb.index = 0
            bpy.ops.mb.pose_add()
            mb.poses[1].library_id = "walk"
            mb.poses[2].library_id = "wave"
            mb.poses[2].armature.location = (0.0, -3.0, 0.0)
            mb.index = 1
            bpy.ops.view3d.view_all()
        state["step"] = 4
        return 1.0
    if s == 4:
        shot("3_poses")
        with bpy.context.temp_override(**c):
            mb.index = 2
            log("edit ->", bpy.ops.mb.limb_edit_begin())
        state["step"] = 5
        return 1.0
    if s == 5:
        shot("4_editing")
        with bpy.context.temp_override(**c):
            log("end ->", bpy.ops.mb.limb_edit_end())
            if CHAR:
                mb.character_path = CHAR
            log("generate ->", bpy.ops.mb.generate(), mb.error)
        state["step"] = 6
        return 1.5
    if s == 6:
        log("playing", c["screen"].is_animation_playing, "frame", c["scene"].frame_current)
        shot("5_generated")
        with bpy.context.temp_override(**c):
            log("stop ->", bpy.ops.mb.stop(), mb.status)
        state["step"] = 7
        return 0.5
    log("DONE")
    bpy.ops.wm.quit_blender()
    return None


bpy.context.preferences.view.show_splash = False
bpy.app.timers.register(tick, first_interval=2.0)
