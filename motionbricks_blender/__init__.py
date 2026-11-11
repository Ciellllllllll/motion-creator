"""MotionBricks を Blender の中で使うアドオン。"""

try:
    import bpy  # noqa: F401
    _IN_BLENDER = True
except ImportError:  # 純 Python のテストから読み込まれた場合
    _IN_BLENDER = False


def register():
    pass


def unregister():
    pass
