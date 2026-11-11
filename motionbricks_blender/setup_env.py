"""MotionBricks のインストールと起動コマンド（bpy 非依存）。

各ステップは完了の印を残すので、途中で失敗しても次回は続きから再開する。
"""

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, NamedTuple

MB_REPO_URL = "https://github.com/NVlabs/motionbricks"
MB_COMMIT = "54fc563aeaaf819a753d72cc9fdc6713989cdd43"
# サンプル動き（評価専用ライセンス）と資料は取得しない。
LFS_EXCLUDE = "assets/motions/**,docs/**"

_cancel = threading.Event()


class SetupError(Exception):
    """利用者に見せる日本語 1 行のエラー。"""


class Step(NamedTuple):
    key: str
    label: str
    weight: float
    done: Callable[[Path], bool]
    run: Callable[[Path, Callable[[str], None]], None]


def mb_home() -> Path:
    override = os.environ.get("MOTIONBRICKS_BLENDER_HOME")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "MotionBricksBlender"


def repo_dir() -> Path:
    return mb_home() / "motionbricks"


def addon_dir() -> Path:
    return Path(__file__).resolve().parent


def find_tool(name: str):
    found = shutil.which(name)
    if found:
        return found
    for cand in (Path.home() / ".local" / "bin" / f"{name}.exe",
                 Path.home() / ".cargo" / "bin" / f"{name}.exe"):
        if cand.exists():
            return str(cand)
    return None


def _run(cmd, cwd, say, env=None):
    """コマンドを窓なしで実行し、出力を 1 行ずつ *say* に流す。"""
    full_env = dict(os.environ)
    full_env.update(env or {})
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=full_env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", creationflags=flags)
    tail = []
    for line in proc.stdout:
        line = line.strip()
        if line:
            tail = (tail + [line])[-5:]
            say(line)
        if _cancel.is_set():
            proc.kill()
            raise SetupError("中止しました")
    if proc.wait() != 0:
        raise RuntimeError(" / ".join(tail[-2:]) or f"終了コード {proc.returncode}")


def _marker(home: Path, key: str) -> Path:
    return home / ".state" / f"{key}.ok"


def _marked(key):
    return lambda home: _marker(home, key).exists()


def _mark(home: Path, key: str):
    m = _marker(home, key)
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(MB_COMMIT, encoding="utf-8")


def _step_tools(home, say):
    missing = [n for n in ("git", "uv") if not find_tool(n)]
    if not missing:
        try:
            _run([find_tool("git"), "lfs", "version"], None, say)
        except Exception:
            missing.append("git-lfs")
    if missing:
        raise RuntimeError("必要なツールがありません: " + ", ".join(missing))
    home.mkdir(parents=True, exist_ok=True)
    _mark(home, "tools")


def _step_clone(home, say):
    dest = home / "motionbricks"
    if dest.exists() and not (dest / ".git").exists():
        shutil.rmtree(dest)
    if not dest.exists():
        _run([find_tool("git"), "clone", "--no-checkout", "--progress", MB_REPO_URL, str(dest)],
             home, say, env={"GIT_LFS_SKIP_SMUDGE": "1"})
    _mark(home, "clone")


def _step_checkout(home, say):
    _run([find_tool("git"), "checkout", "--force", MB_COMMIT], home / "motionbricks", say,
         env={"GIT_LFS_SKIP_SMUDGE": "1"})
    _mark(home, "checkout")


def _step_lfs(home, say):
    _run([find_tool("git"), "lfs", "pull", "--exclude", LFS_EXCLUDE], home / "motionbricks", say)
    _mark(home, "lfs")


def _step_sync(home, say):
    _run([find_tool("uv"), "sync", "--locked"], home / "motionbricks", say)
    _mark(home, "sync")


def _sync_done(home):
    return _marker(home, "sync").exists() and (home / "motionbricks" / ".venv").exists()


STEPS = [
    Step("tools", "ツールを確認中", 0.02, _marked("tools"), _step_tools),
    Step("clone", "MotionBricksを取得中", 0.13, _marked("clone"), _step_clone),
    Step("checkout", "バージョンを固定中", 0.05, _marked("checkout"), _step_checkout),
    Step("lfs", "モデルを取得中", 0.30, _marked("lfs"), _step_lfs),
    Step("sync", "環境を準備中", 0.50, _sync_done, _step_sync),
]


def is_installed() -> bool:
    home = mb_home()
    return all(step.done(home) for step in STEPS)


def install(progress: Callable[[float, str], None], cancel: threading.Event) -> None:
    """未完了のステップだけを順に実行する。失敗は SetupError（日本語）。"""
    global _cancel
    _cancel = cancel
    home = mb_home()
    home.mkdir(parents=True, exist_ok=True)
    total = sum(s.weight for s in STEPS) or 1.0
    acc = 0.0
    for step in STEPS:
        if cancel.is_set():
            raise SetupError("中止しました")
        frac = acc / total
        if not step.done(home):
            progress(frac, step.label)
            try:
                step.run(home, lambda line, f=frac, lab=step.label: progress(f, lab))
            except SetupError:
                raise
            except Exception as e:
                raise SetupError(f"{step.label}に失敗しました: {e}") from e
        acc += step.weight
    progress(1.0, "準備完了")


def server_command(port: int) -> list:
    uv = find_tool("uv") or "uv"
    return [uv, "run", "--project", str(repo_dir()), "python",
            str(addon_dir() / "server" / "mb_server.py"), "--port", str(port)]
