"""パネルと、MotionBricks の起動・停止。"""

import atexit
import threading

import bpy
from bpy.app.handlers import persistent

from . import client as mb_client
from . import export, limb_edit, poses, setup_env
from .client import Client, ClientError
from .setup_env import SetupError

CATEGORY = "MotionBricks"
_icons = {}
_INSTALL_SHARE = 0.7        # 進み具合のうち、インストールが占める割合

# サーバーは Blender 1 つにつき 1 つ。ファイルを開き直しても動かしたままにする。
_server = {"proc": None, "port": 0}
_job = {"thread": None, "cancel": None, "state": None}
_msgbus_owner = object()


# ---- サーバー ---------------------------------------------------------------

def server_process():
    return _server["proc"]


def server_alive() -> bool:
    proc = _server["proc"]
    if proc is None or proc.poll() is not None:
        return False
    try:
        return bool(Client(_server["port"]).ping(2.0).get("ready"))
    except ClientError:
        return False


def stop_server() -> None:
    proc, port = _server["proc"], _server["port"]
    _server["proc"], _server["port"] = None, 0
    if proc is None:
        return
    if proc.poll() is None:
        try:
            Client(port).shutdown()
            proc.wait(timeout=5)
        except Exception:
            pass
    mb_client.kill_tree(proc)


def _work(state, cancel):
    """別スレッドで動く：インストール → サーバー起動。bpy には触らない。"""
    try:
        if server_alive():
            state.update(progress=1.0, label="起動中")
        else:
            stop_server()
            if not setup_env.is_installed():
                state["status"] = 'INSTALLING'
                setup_env.install(lambda f, t: state.update(progress=f * _INSTALL_SHARE, label=t), cancel)
            state.update(status='STARTING', label="モデルを準備中", progress=_INSTALL_SHARE)
            ticks = [0]

            def on_wait(_listening):
                ticks[0] += 1    # 初回は TensorRT の準備に 1〜2 分かかる。少しずつ進める
                state["progress"] = _INSTALL_SHARE + (1.0 - _INSTALL_SHARE) * (1.0 - 0.98 ** ticks[0])

            proc, port = mb_client.start_server(on_wait=on_wait, cancel=cancel)
            _server["proc"], _server["port"] = proc, port
        state["skeleton"] = Client(_server["port"]).skeleton()
        state["done"] = True
    except (SetupError, ClientError) as e:
        state["error"] = str(e)
    except Exception as e:   # 想定外でも、利用者には日本語 1 行で伝える
        state["error"] = f"起動に失敗しました（{type(e).__name__}）"


def _apply_progress(mb, state):
    mb.status = state.get("status", mb.status)
    mb.progress = min(1.0, max(0.0, state.get("progress", 0.0)))
    mb.progress_label = state.get("label", "")


def _finish(scene, state):
    mb = scene.mb
    mb.busy = False
    if state.get("error"):
        if state["error"] == "中止しました":
            mb.status, mb.error = 'STOPPED', ""
        else:
            mb.status, mb.error = 'ERROR', state["error"]
        return False
    mb.server_port = _server["port"]
    mb.progress, mb.progress_label = 1.0, ""
    try:
        poses.on_server_started(scene, state["skeleton"])
    except ClientError as e:
        mb.status, mb.error = 'ERROR', str(e)
        return False
    return True


def _redraw():
    wm = bpy.context.window_manager
    for window in wm.windows if wm else ():
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class MB_OT_start(bpy.types.Operator):
    bl_idname = "mb.start"
    bl_label = "MotionBricksを起動"

    @classmethod
    def poll(cls, context):
        return context.scene.mb.status in {'STOPPED', 'ERROR'} and _job["thread"] is None

    def execute(self, context):
        scene = context.scene
        mb = scene.mb
        mb.error = ""
        mb.status, mb.progress, mb.progress_label = 'STARTING', 0.0, "起動中"
        state = {"status": 'STARTING', "progress": 0.0, "label": "起動中"}
        cancel = threading.Event()
        if bpy.app.background:          # 画面なし（テスト）ではその場で待つ
            _work(state, cancel)
            return {'FINISHED'} if _finish(scene, state) else {'CANCELLED'}
        thread = threading.Thread(target=_work, args=(state, cancel), daemon=True)
        _job.update(thread=thread, cancel=cancel, state=state)
        scene_name = scene.name
        thread.start()

        def poll_job():
            target = bpy.data.scenes.get(scene_name)
            if target is None:
                _job.update(thread=None, cancel=None, state=None)
                return None
            if thread.is_alive():
                _apply_progress(target.mb, state)
                _redraw()
                return 0.2
            _job.update(thread=None, cancel=None, state=None)
            _finish(target, state)
            _redraw()
            return None

        bpy.app.timers.register(poll_job, first_interval=0.2)
        return {'FINISHED'}


class MB_OT_cancel(bpy.types.Operator):
    bl_idname = "mb.cancel"
    bl_label = "中止"

    def execute(self, context):
        if _job["cancel"] is not None:
            _job["cancel"].set()
        return {'FINISHED'}


class MB_OT_stop(bpy.types.Operator):
    bl_idname = "mb.stop"
    bl_label = "停止"

    def execute(self, context):
        limb_edit.finish_if_active(context.scene)
        stop_server()
        mb = context.scene.mb
        mb.status, mb.error, mb.server_port = 'STOPPED', "", 0
        return {'FINISHED'}


# ---- 描画 -------------------------------------------------------------------

def _big(layout, scale=1.5):
    row = layout.row()
    row.scale_y = scale
    return row


def draw_main(layout, context):
    mb = context.scene.mb
    if mb.status in {'INSTALLING', 'STARTING'}:
        layout.progress(factor=mb.progress, text=f"{mb.progress_label or '起動中'}… {round(mb.progress * 100)}%")
        layout.operator("mb.cancel", text="中止", icon='CANCEL')
        return
    if mb.status == 'RUNNING':
        row = layout.row(align=True)
        row.label(text="起動中", icon='CHECKMARK')
        row.operator("mb.stop", text="停止", icon='SNAP_FACE')
        if mb.error:
            warn = layout.row()
            warn.alert = True
            warn.label(text=mb.error, icon='ERROR')
        return
    if mb.status == 'ERROR' and mb.error:
        warn = layout.row()
        warn.alert = True
        warn.label(text=mb.error, icon='ERROR')
    _big(layout, 1.8).operator("mb.start", text="MotionBricksを起動", icon='PLAY')


def kind_icons() -> dict:
    """一覧の色つきアイコン（緑・オレンジ・青）。Blender の版によってアイコン名が違う。"""
    if not _icons:
        valid = set(bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys())
        for kind, number, fallback in (('START', "04", 'KEYTYPE_JITTER_VEC'), ('MIDDLE', "02", 'KEYTYPE_EXTREME_VEC'),
                                       ('END', "05", 'KEYTYPE_BREAKDOWN_VEC')):
            names = (f"STRIP_COLOR_{number}", f"SEQUENCE_COLOR_{number}", fallback)
            _icons[kind] = next((n for n in names if n in valid), 'DOT')
    return _icons


class MB_UL_poses(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=poses.entry_label(data, index), icon=kind_icons()[item.kind])


def draw_poses(layout, context):
    mb = context.scene.mb
    editing = mb.edit_index >= 0
    row = layout.row()
    listing = row.column()
    listing.enabled = not editing
    listing.template_list("MB_UL_poses", "", mb, "poses", mb, "index", rows=4)
    side = row.column(align=True)
    side.enabled = not editing
    side.operator("mb.pose_add", text="", icon='ADD')
    side.operator("mb.pose_remove", text="", icon='REMOVE')
    side.separator()
    side.operator("mb.pose_move", text="", icon='TRIA_UP').direction = 'UP'
    side.operator("mb.pose_move", text="", icon='TRIA_DOWN').direction = 'DOWN'
    if not (0 <= mb.index < len(mb.poses)):
        return
    entry = mb.poses[mb.index]
    col = layout.column()
    col.use_property_split = True
    col.use_property_decorate = False
    col.enabled = not editing
    col.prop(entry, "library_id", text="ポーズ集")
    col.prop(entry, "time", text="タイミング", slider=True)
    row = layout.row(align=True)
    if editing:
        row.operator("mb.limb_edit_end", text="編集を終える", icon='CHECKMARK', depress=True)
    else:
        row.operator("mb.limb_edit_begin", text="手足を編集", icon='VIEW_PAN')
    row.operator("mb.limb_edit_reset", text="元に戻す", icon='LOOP_BACK')


def draw_generate(layout, context):
    _big(layout, 1.8).operator("mb.generate", text="生成", icon='PLAY')
    row = layout.row(align=True)
    playing = bool(context.screen and context.screen.is_animation_playing)
    row.operator("mb.play", text="再生", icon='PAUSE' if playing else 'TRIA_RIGHT', depress=playing)
    row.operator("mb.regenerate", text="もう一度", icon='FILE_REFRESH')


def draw_export(layout, context):
    mb = context.scene.mb
    layout.row(align=True).prop(mb, "export_mode", expand=True)
    col = layout.column(align=True)
    col.label(text="キャラクター", icon='ARMATURE_DATA')
    row = col.row()
    row.alert = not mb.character_path
    row.prop(mb, "character_path", text="")
    layout.prop(mb, "in_place", text="In Place", icon='PINNED', toggle=True)
    _big(layout, 1.8).operator("mb.export", text="Mixamo形式で保存", icon='EXPORT')


class _Panel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = CATEGORY


class MB_PT_main(_Panel, bpy.types.Panel):
    bl_idname = "MB_PT_main"
    bl_label = "MotionBricks"
    bl_order = 0

    def draw(self, context):
        draw_main(self.layout, context)


class _RunningPanel(_Panel):
    @classmethod
    def poll(cls, context):
        return context.scene.mb.status == 'RUNNING'


class MB_PT_poses(_RunningPanel, bpy.types.Panel):
    bl_idname = "MB_PT_poses"
    bl_label = "ポーズ"
    bl_order = 1

    def draw(self, context):
        draw_poses(self.layout, context)


class MB_PT_generate(_RunningPanel, bpy.types.Panel):
    bl_idname = "MB_PT_generate"
    bl_label = "生成"
    bl_order = 2

    def draw(self, context):
        draw_generate(self.layout, context)


class MB_PT_export(_RunningPanel, bpy.types.Panel):
    bl_idname = "MB_PT_export"
    bl_label = "書き出し"
    bl_order = 3

    def draw(self, context):
        draw_export(self.layout, context)


# ---- 3D ビューで人形を選んだら、一覧の選択も合わせる ---------------------------

def _on_active_object_changed():
    context = bpy.context
    scene = getattr(context, "scene", None)
    mb = getattr(scene, "mb", None)
    if mb is None or mb.edit_index >= 0:
        return
    active = context.view_layer.objects.active if context.view_layer else None
    for i, entry in enumerate(mb.poses):
        if entry.armature is not None and entry.armature == active and mb.index != i:
            poses.set_index_quietly(mb, i)
            _redraw()
            return


def _subscribe():
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    bpy.msgbus.subscribe_rna(key=(bpy.types.LayerObjects, "active"), owner=_msgbus_owner, args=(),
                             notify=_on_active_object_changed)


@persistent
def _on_load_post(*_):
    _subscribe()


classes = (MB_OT_start, MB_OT_cancel, MB_OT_stop, MB_UL_poses, MB_PT_main, MB_PT_poses, MB_PT_generate,
           MB_PT_export)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.app.handlers.load_post.append(_on_load_post)
    _subscribe()
    atexit.register(stop_server)


def unregister():
    if _job["cancel"] is not None:
        _job["cancel"].set()
    stop_server()
    atexit.unregister(stop_server)
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
