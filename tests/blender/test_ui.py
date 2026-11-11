import socket
import time

import bpy

import common
import motionbricks_blender
from motionbricks_blender import poses, ui


def teardown():
    poses.set_client_factory(None)


def teardown_module():
    ui.stop_server()


class Layout:
    """描画の呼び出しを記録する偽のレイアウト。"""

    def __init__(self, log=None):
        self.log = log if log is not None else []
        self.alert = self.enabled = self.active = False
        self.scale_x = self.scale_y = 1.0

    def _child(self, *a, **k):
        return Layout(self.log)

    row = column = box = split = column_flow = grid_flow = _child

    def label(self, text="", icon='NONE', **k):
        self.log.append(("label", text))

    def operator(self, idname, text=None, icon='NONE', depress=False, **k):
        assert text is not None, f"{idname} にラベルがない"
        self.log.append(("op", text, idname))
        return type("Props", (), {})()

    def prop(self, data, name, text=None, icon='NONE', expand=False, **k):
        prop = data.bl_rna.properties[name]
        if expand and prop.type == 'ENUM':
            for item in prop.enum_items:
                self.log.append(("prop", item.name, name))
        else:
            self.log.append(("prop", prop.name if text is None else text, name))

    def template_list(self, list_type, list_id, data, prop, active_data, active_prop, **k):
        mb = bpy.context.scene.mb
        for i in range(len(mb.poses)):
            self.log.append(("list", poses.entry_label(mb, i)))

    def progress(self, factor=0.0, text="", **k):
        self.log.append(("progress", text))

    def separator(self, **k):
        pass


def _texts(draw):
    layout = Layout()
    draw(layout, bpy.context)
    return [entry[1] for entry in layout.log if entry[1]]


def _all_texts():
    return [_texts(ui.draw_main), _texts(ui.draw_poses), _texts(ui.draw_generate), _texts(ui.draw_export)]


def test_stopped_shows_only_launch_button():
    mb = bpy.context.scene.mb
    assert mb.status == 'STOPPED'
    assert _texts(ui.draw_main) == ["MotionBricksを起動"]
    assert all(cls.poll(bpy.context) is False for cls in (ui.MB_PT_poses, ui.MB_PT_generate, ui.MB_PT_export))
    assert ui.MB_PT_main.bl_category == "MotionBricks"


def test_installing_shows_progress_and_cancel():
    mb = bpy.context.scene.mb
    mb.status, mb.progress, mb.progress_label = 'INSTALLING', 0.45, "モデルを取得中"
    assert _texts(ui.draw_main) == ["モデルを取得中… 45%", "中止"]
    mb.status = 'STARTING'
    assert _texts(ui.draw_main)[-1] == "中止"
    assert ui.MB_PT_poses.poll(bpy.context) is False


def test_error_shows_message_and_retry():
    mb = bpy.context.scene.mb
    mb.status, mb.error = 'ERROR', "MotionBricksに接続できません"
    assert _texts(ui.draw_main) == ["MotionBricksに接続できません", "MotionBricksを起動"]


def test_running_panel_order_and_labels():
    common.start_fake()
    mb = bpy.context.scene.mb
    mb.index = 0
    bpy.ops.mb.pose_add()
    main, pose, gen, exp = _all_texts()
    assert main == ["起動中", "停止"]
    assert pose == ["最初", "途中 1", "最後", "ポーズ集", "タイミング", "手足を編集", "元に戻す"]
    assert gen == ["生成", "再生", "もう一度"]
    assert exp == ["キャラに焼き込み", "動きのみ", "キャラクター", "In Place", "Mixamo形式で保存"]
    assert all(cls.poll(bpy.context) for cls in (ui.MB_PT_poses, ui.MB_PT_generate, ui.MB_PT_export))
    assert [ui.MB_PT_main.bl_label, ui.MB_PT_poses.bl_label, ui.MB_PT_generate.bl_label,
            ui.MB_PT_export.bl_label] == ["MotionBricks", "ポーズ", "生成", "書き出し"]
    allowed = {"In Place", "MotionBricks"}
    for text in main + pose + gen + exp:
        assert text in allowed or any(ord(ch) > 0x3000 for ch in text), text      # 英語だけのラベルは無い


def test_editing_shows_finish_button():
    common.start_fake()
    mb = bpy.context.scene.mb
    mb.index = 1
    bpy.ops.mb.limb_edit_begin()
    assert "編集を終える" in _texts(ui.draw_poses) and "手足を編集" not in _texts(ui.draw_poses)


def test_warning_is_shown_while_running():
    common.start_fake()
    bpy.context.scene.mb.error = "Mixamoのキャラクターではありません"
    assert _texts(ui.draw_main) == ["起動中", "停止", "Mixamoのキャラクターではありません"]


def test_operator_labels_are_japanese():
    for name in dir(bpy.types):
        if name.startswith("MB_OT_"):
            label = getattr(bpy.types, name).bl_label
            assert any(ord(ch) > 0x3000 for ch in label), (name, label)


def _port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def test_start_stop_real_server():
    mb = bpy.context.scene.mb
    t0 = time.time()
    assert bpy.ops.mb.start() == {'FINISHED'}, mb.error
    elapsed = time.time() - t0
    assert mb.status == 'RUNNING' and mb.server_port > 0 and len(mb.poses) == 2
    assert elapsed < 60, elapsed                                  # 2 回目以降の起動は 1 分以内
    port = mb.server_port
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error
    assert bpy.ops.mb.stop() == {'FINISHED'}
    assert mb.status == 'STOPPED'
    deadline = time.time() + 10
    while _port_open(port) and time.time() < deadline:
        time.sleep(0.3)
    assert not _port_open(port)
    assert len(mb.poses) == 2                                     # 止めてもポーズは残る


def test_server_killed_on_disable():
    mb = bpy.context.scene.mb
    assert bpy.ops.mb.start() == {'FINISHED'}, mb.error
    port = mb.server_port
    motionbricks_blender.unregister()
    try:
        deadline = time.time() + 10
        while _port_open(port) and time.time() < deadline:
            time.sleep(0.3)
        assert not _port_open(port)
    finally:
        motionbricks_blender.register()


def test_restart_after_server_death():
    mb = bpy.context.scene.mb
    assert bpy.ops.mb.start() == {'FINISHED'}, mb.error
    first = mb.server_port
    from motionbricks_blender import client
    client.kill_tree(ui.server_process())
    ui.server_process().wait(timeout=15)
    try:
        bpy.ops.mb.generate()
    except RuntimeError:
        pass
    assert mb.status == 'ERROR'
    assert bpy.ops.mb.start() == {'FINISHED'}, mb.error
    assert mb.status == 'RUNNING' and mb.server_port != first and mb.error == ""
    assert bpy.ops.mb.generate() == {'FINISHED'}, mb.error


def test_start_reuses_running_server_after_file_load():
    mb = bpy.context.scene.mb
    assert bpy.ops.mb.start() == {'FINISHED'}, mb.error
    port = mb.server_port
    bpy.ops.wm.read_homefile(use_empty=True, use_factory_startup=True)
    mb = bpy.context.scene.mb
    assert mb.status == 'STOPPED'
    t0 = time.time()
    assert bpy.ops.mb.start() == {'FINISHED'}, mb.error
    assert mb.server_port == port and time.time() - t0 < 5
