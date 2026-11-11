"""実際の画面つき Blender で、長く使ったときの確認をする（連続操作）。

blender --factory-startup --python tools/gui_soak.py -- <出力フォルダ> <キャラクター FBX> [回数]

毎回：ポーズ集と配置をランダムに変える → 手足を編集 → 生成。ときどき、途中の追加・削除、
もう一度、元に戻す／やり直す、保存して開き直す、書き出しを行う。結果は soak.log に残す。
"""
import math
import os
import random
import subprocess
import sys
import time
import traceback

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
OUT, CHAR = args[0], args[1]
ROUNDS = int(args[2]) if len(args) > 2 else 30
os.makedirs(OUT, exist_ok=True)
MODULE = "bl_ext.user_default.motionbricks_blender"
LIB = ["base", "stand", "walk", "jog", "run", "crouch", "crouch_walk", "jump", "sit", "wave", "turn"]
HANDLES = ["MB_左手", "MB_右手", "MB_左足", "MB_右足", "MB_左ひじ", "MB_右ひざ"]
rnd = random.Random(7)
S = {"phase": "enable", "round": 0, "t0": time.time(), "errors": 0, "gen_ms": [], "exports": 0, "reloads": 0,
     "undos": 0, "wait": 0}


def log(*a):
    with open(os.path.join(OUT, "soak.log"), "a", encoding="utf-8") as f:
        f.write(" ".join(str(x) for x in a) + "\n")


def gpu_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def ctx():
    window = bpy.context.window_manager.windows[0]
    area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return {"window": window, "screen": window.screen, "area": area, "region": region, "scene": window.scene}


def counts():
    return (f"objects {len(bpy.data.objects)} actions {len(bpy.data.actions)} armatures {len(bpy.data.armatures)} "
            f"meshes {len(bpy.data.meshes)} gpu {gpu_mb()} MB")


def check_result(mb, scene):
    act = mb.result_action
    assert act is not None, "結果がない"
    arm = bpy.data.objects["MB_結果"]
    end = int(act.frame_range[1])
    assert 18 <= end <= 2000, end
    for f in (1, end // 2, end):
        scene.frame_set(f)
        p = arm.matrix_world @ arm.pose.bones["Hips"].head
        assert all(math.isfinite(v) for v in p) and 0.2 < p.z < 2.2, (f, tuple(p))
    return end


def step():
    c = ctx()
    scene = c["scene"]
    mb = getattr(scene, "mb", None)
    if time.time() - S["t0"] > 1500:
        log("TIMEOUT", S["phase"], S["round"])
        return finish()
    ph = S["phase"]
    with bpy.context.temp_override(**c):
        if ph == "enable":
            bpy.ops.preferences.addon_enable(module=MODULE)
            c["area"].spaces[0].show_region_ui = True
            for o in list(bpy.data.objects):
                bpy.data.objects.remove(o)
            log("START", counts())
            S["phase"] = "start"
            return 0.5
        if ph == "start":
            if mb.status in {'STOPPED', 'ERROR'} and bpy.ops.mb.start.poll():
                bpy.ops.mb.start()
            S["phase"] = "wait_start"
            return 0.3
        if ph == "wait_start":
            if mb.status in {'INSTALLING', 'STARTING'}:
                return 0.3
            assert mb.status == 'RUNNING', (mb.status, mb.error)
            if not mb.character_path:
                mb.character_path = CHAR
            S["phase"] = "arrange"
            return 0.2
        if ph == "arrange":
            S["round"] += 1
            r = S["round"]
            if r > ROUNDS:
                return finish()
            middles = sum(1 for e in mb.poses if e.kind == 'MIDDLE')
            if middles < 3 and rnd.random() < 0.5:
                mb.index = rnd.randrange(len(mb.poses))
                bpy.ops.mb.pose_add()
            elif middles > 0 and rnd.random() < 0.3:
                mb.index = next(i for i, e in enumerate(mb.poses) if e.kind == 'MIDDLE')
                bpy.ops.mb.pose_remove()
            if middles >= 2 and rnd.random() < 0.3:
                mb.index = next(i for i, e in enumerate(mb.poses) if e.kind == 'MIDDLE')
                if bpy.ops.mb.pose_move.poll():
                    bpy.ops.mb.pose_move(direction='DOWN')
            y = 0.0
            for i, e in enumerate(mb.poses):
                mb.index = i                      # 隠した人形を出し直す
                e.library_id = rnd.choice(LIB)
                e.time = rnd.random()
                if i > 0:
                    y -= rnd.uniform(0.8, 3.5)
                e.armature.location = (rnd.uniform(-1.5, 1.5), y, 0.0)
                e.armature.rotation_euler.z = rnd.uniform(-1.0, 1.0)
            mb.index = rnd.randrange(len(mb.poses))
            if rnd.random() < 0.7 and bpy.ops.mb.limb_edit_begin.poll():
                bpy.ops.mb.limb_edit_begin()
                arm = mb.poses[mb.index].armature
                for name in rnd.sample(HANDLES, 3):
                    pb = arm.pose.bones[name]
                    delta = Vector((rnd.uniform(-0.25, 0.25), rnd.uniform(-0.25, 0.25), rnd.uniform(-0.1, 0.35)))
                    pb.location = pb.location + arm.data.bones[name].matrix_local.to_3x3().inverted() @ delta
                S["phase"] = "edited"
                return 0.3                        # 画面の更新（解き直し）を待つ
            S["phase"] = "generate"
            return 0.05
        if ph == "edited":
            if rnd.random() < 0.5:
                bpy.ops.mb.limb_edit_end()        # 半分は「生成」に自動で確定させる
            S["phase"] = "generate"
            return 0.05
        if ph == "generate":
            r = S["round"]
            t = time.time()
            result = bpy.ops.mb.generate()
            S["gen_ms"].append((time.time() - t) * 1000)
            assert result == {'FINISHED'} and not mb.error, (result, mb.error)
            end = check_result(mb, scene)
            assert mb.edit_index == -1
            assert all(len(e.armature.data.bones) == 77 for e in mb.poses), "ハンドルが残っている"
            extra = ""
            if r % 5 == 0:
                bpy.ops.mb.regenerate()
                extra += f" regen->{check_result(mb, scene)}"
            if r % 6 == 0:
                mb.export_mode = rnd.choice(['BAKE', 'MOTION'])
                mb.in_place = rnd.random() < 0.5
                path = os.path.join(OUT, f"soak_{r:02d}.fbx")
                assert bpy.ops.mb.export(filepath=path) == {'FINISHED'}, mb.error
                assert os.path.getsize(path) > 50000
                S["exports"] += 1
                extra += " export"
            log(f"ROUND {r} poses {len(mb.poses)} frames {end} gen {S['gen_ms'][-1]:.0f} ms{extra} | {counts()}")
            if r % 7 == 0:
                S["phase"] = "undo"
            elif r % 10 == 0:
                S["phase"] = "save"
            else:
                S["phase"] = "arrange"
            return 0.3
        if ph == "undo":
            try:
                for _ in range(3):
                    bpy.ops.ed.undo()
                for _ in range(2):
                    bpy.ops.ed.redo()
                S["undos"] += 1
                log("  undo x3 / redo x2 ok:", len(scene.mb.poses), "poses, status", scene.mb.status)
            except RuntimeError as e:
                log("  undo not available:", str(e).strip()[:80])
            S["phase"] = "after_undo"
            return 0.3
        if ph == "after_undo":
            assert mb.status == 'RUNNING' and len(mb.poses) >= 2, (mb.status, len(mb.poses))
            result = bpy.ops.mb.generate()        # 元に戻した後も続けて使える
            assert result == {'FINISHED'} and not mb.error, (result, mb.error)
            check_result(mb, scene)
            S["phase"] = "save" if S["round"] % 10 == 0 else "arrange"
            return 0.2
        if ph == "save":
            path = os.path.join(OUT, "作業中.blend")
            bpy.ops.wm.save_as_mainfile(filepath=path)
            bpy.ops.wm.open_mainfile(filepath=path)
            S["reloads"] += 1
            S["phase"] = "reloaded"
            return 0.5
        if ph == "reloaded":
            assert mb.status == 'STOPPED' and len(mb.poses) >= 2 and mb.character_path, (mb.status, len(mb.poses))
            log("  reloaded:", len(mb.poses), "poses, result kept:", mb.result_action is not None)
            S["phase"] = "start"
            return 0.2
    return 0.2


def finish():
    c = ctx()
    with bpy.context.temp_override(**c):
        mb = c["scene"].mb
        try:
            bpy.ops.mb.stop()
        except Exception:
            pass
    ms = sorted(S["gen_ms"]) or [0]
    log(f"SUMMARY rounds {min(S['round'], ROUNDS)} errors {S['errors']} exports {S['exports']} reloads {S['reloads']} "
        f"undos {S['undos']} gen median {ms[len(ms) // 2]:.0f} ms max {ms[-1]:.0f} ms | {counts()} | "
        f"{time.time() - S['t0']:.0f} s")
    bpy.ops.wm.quit_blender()
    return None


def tick():
    try:
        return step()
    except Exception:
        S["errors"] += 1
        log("EXCEPTION in", S["phase"], "round", S["round"], "\n" + traceback.format_exc())
        if S["errors"] >= 3:
            return finish()
        S["phase"] = "arrange" if S["phase"] not in {"enable", "start", "wait_start"} else S["phase"]
        try:
            from bl_ext.user_default.motionbricks_blender import limb_edit
            limb_edit.finish_if_active(bpy.context.window_manager.windows[0].scene)
        except Exception:
            pass
        return 0.5


bpy.context.preferences.view.show_splash = False
bpy.app.timers.register(tick, first_interval=2.0, persistent=True)
