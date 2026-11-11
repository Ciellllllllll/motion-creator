"""Blender 内でテストを実行する。

blender -b --factory-startup --python tests/blender/run.py -- <test_module> [test_name ...]
"""
import importlib
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "blender"))


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    module_name, only = args[0], set(args[1:])
    import bpy
    import motionbricks_blender
    motionbricks_blender.register()
    failed = []
    try:
        module = importlib.import_module(module_name)
        tests = [(n, f) for n, f in vars(module).items() if n.startswith("test_") and callable(f)]
        for name, fn in tests:
            if only and name not in only:
                continue
            bpy.ops.wm.read_factory_settings(use_empty=True)
            try:
                if hasattr(module, "setup"):
                    module.setup()
                fn()
                print(f"PASS {name}", flush=True)
            except BaseException:
                failed.append(name)
                print(f"FAIL {name}", flush=True)
                traceback.print_exc()
            finally:
                if hasattr(module, "teardown"):
                    try:
                        module.teardown()
                    except Exception:
                        traceback.print_exc()
    except BaseException:
        traceback.print_exc()
        failed.append("<import>")
    finally:
        try:
            motionbricks_blender.unregister()
        except Exception:
            traceback.print_exc()
    print(f"RESULT {module_name}: {'FAILED ' + ', '.join(failed) if failed else 'ALL PASSED'}", flush=True)
    sys.stdout.flush()
    import os
    os._exit(1 if failed else 0)


main()
