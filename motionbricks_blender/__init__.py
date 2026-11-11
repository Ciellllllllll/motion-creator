"""MotionBricks を Blender の中で使うアドオン。"""

_MODULE_NAMES = ("poses", "limb_edit", "export", "generate", "ui", "overlay")


def _modules():
    import importlib
    return [importlib.import_module(f".{name}", __name__) for name in _MODULE_NAMES]


def register():
    for module in _modules():
        module.register()


def unregister():
    for module in reversed(_modules()):
        module.unregister()
