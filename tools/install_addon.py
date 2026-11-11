"""アドオンを Blender のユーザー拡張機能フォルダへコピーして有効にする。

blender -b --python tools/install_addon.py
"""
import pathlib
import shutil

import bpy

SRC = pathlib.Path(__file__).resolve().parents[1] / "motionbricks_blender"
MODULE = "bl_ext.user_default.motionbricks_blender"
dest = pathlib.Path(bpy.utils.user_resource('EXTENSIONS', path="user_default", create=True)) / "motionbricks_blender"
if dest.exists():
    try:
        bpy.ops.preferences.addon_disable(module=MODULE)
    except Exception:
        pass
    shutil.rmtree(dest)
shutil.copytree(SRC, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
bpy.ops.extensions.repo_refresh_all() if hasattr(bpy.ops.extensions, "repo_refresh_all") else None
bpy.ops.preferences.addon_enable(module=MODULE)
bpy.ops.wm.save_userpref()
enabled = MODULE in bpy.context.preferences.addons
print("INSTALLED", dest, "enabled:", enabled, "panel:", hasattr(bpy.types, "MB_PT_main"))
