"""Mixamo キャラクターの読み込みと FBX の書き出し。"""

import os

import bpy

from . import rig_transfer as rt

CHAR_COLLECTION = "MB_キャラクター"
PATH_KEY = "mb_char_path"
MTIME_KEY = "mb_char_mtime"


def _collection(scene):
    coll = bpy.data.collections.get(CHAR_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(CHAR_COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def character_armature(scene):
    coll = bpy.data.collections.get(CHAR_COLLECTION)
    if coll is None:
        return None
    return next((o for o in coll.objects if o.type == 'ARMATURE' and PATH_KEY in o), None)


def character_objects(scene):
    coll = bpy.data.collections.get(CHAR_COLLECTION)
    return list(coll.objects) if coll else []


def _delete(objects):
    datas = [o.data for o in objects if o.data is not None]
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for data in datas:
        if data.users == 0:
            try:
                bpy.data.batch_remove([data])
            except Exception:
                pass


def remove_character(scene):
    _delete(character_objects(scene))


def ensure_character(scene, path: str):
    """Mixamo キャラクターの FBX を読み込む（同じファイルなら読み直さない）。元のファイルは変更しない。"""
    full = os.path.normpath(bpy.path.abspath(path))
    if not os.path.isfile(full):
        raise ValueError("キャラクターのファイルが見つかりません")
    mtime = os.path.getmtime(full)
    existing = character_armature(scene)
    if existing is not None and existing[PATH_KEY] == full and abs(existing[MTIME_KEY] - mtime) < 1e-3:
        return existing

    context = bpy.context
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    view_layer = context.view_layer
    prev_active = view_layer.objects.active
    prev_selected = [o for o in view_layer.objects if o.select_get()]
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.fbx(filepath=full)
    except Exception as e:
        _delete([o for o in bpy.data.objects if o not in before])
        raise ValueError("キャラクターを読み込めませんでした") from e
    new = [o for o in bpy.data.objects if o not in before]
    arm = next((o for o in new if o.type == 'ARMATURE' and rt.detect_mixamo_prefix(o) is not None), None)
    if arm is None:
        _delete(new)
        raise rt.NotMixamoError("Mixamoのキャラクターではありません")

    remove_character(scene)
    coll = _collection(scene)
    for obj in new:
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        coll.objects.link(obj)
        obj.select_set(False)
    arm[PATH_KEY] = full
    arm[MTIME_KEY] = mtime
    for obj in prev_selected:
        try:
            obj.select_set(True)
        except (RuntimeError, ReferenceError):
            pass
    try:
        view_layer.objects.active = prev_active
    except ReferenceError:
        pass
    return arm


classes = ()


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
