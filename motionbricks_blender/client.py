"""MotionBricks サーバーとの通信と起動（bpy 非依存）。"""

import os
import socket
import subprocess
import time

from . import protocol, setup_env


class ClientError(Exception):
    """利用者に見せる日本語 1 行のエラー。"""


class Client:
    def __init__(self, port: int, timeout: float = 120.0):
        self.port = port
        self.timeout = timeout

    def _call(self, payload: dict, timeout=None) -> dict:
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=timeout or self.timeout) as s:
                s.settimeout(timeout or self.timeout)
                protocol.send(s, payload)
                reply = protocol.recv(s)
        except socket.timeout as e:
            raise ClientError("MotionBricksの応答がありません") from e
        except (OSError, ConnectionError, ValueError) as e:
            raise ClientError("MotionBricksに接続できません") from e
        if not reply.get("ok"):
            raise ClientError(reply.get("error") or "MotionBricksでエラーが起きました")
        return reply

    def ping(self, timeout: float = 5.0) -> dict:
        return self._call({"cmd": "ping"}, timeout)

    def skeleton(self) -> dict:
        r = self._call({"cmd": "skeleton"})
        return {k: r[k] for k in ("joint_names", "parents", "rest", "model_joints")}

    def sample(self, clip: str, time_s: float, anchor: int, placement, static: bool = False) -> list:
        return self._call({"cmd": "sample", "clip": clip, "time": time_s, "anchor": anchor,
                           "placement": list(placement), "static": bool(static)})["poses"]

    def generate(self, context, targets, token_counts) -> dict:
        r = self._call({"cmd": "generate", "context": context, "targets": targets,
                        "token_counts": token_counts})
        return {k: r[k] for k in ("frames", "contacts", "segments")}

    def fk(self, pose) -> list:
        return self._call({"cmd": "fk", "pose": pose})["world"]

    def shutdown(self) -> None:
        self._call({"cmd": "shutdown"}, 5.0)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def log_path():
    return setup_env.mb_home() / "server.log"


def kill_tree(proc) -> None:
    """uv 経由で起動した子プロセスごと止める。"""
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        proc.kill()


def start_server(port=None, on_wait=None, cancel=None, timeout: float = 900.0):
    """サーバーを起動し、生成できる状態になるまで待つ。(Popen, port) を返す。"""
    port = port or free_port()
    cmd = setup_env.server_command(port) + ["--parent-pid", str(os.getpid())]
    log = open(log_path(), "w", encoding="utf-8", errors="replace")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    env.pop("VIRTUAL_ENV", None)
    proc = subprocess.Popen(cmd, cwd=str(setup_env.repo_dir()), stdout=log, stderr=subprocess.STDOUT, env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    client = Client(port)
    deadline = time.time() + timeout
    while True:
        if cancel is not None and cancel.is_set():
            kill_tree(proc)
            raise ClientError("中止しました")
        if proc.poll() is not None:
            raise ClientError("MotionBricksを起動できませんでした")
        if time.time() > deadline:
            kill_tree(proc)
            raise ClientError("MotionBricksの起動に時間がかかりすぎています")
        try:
            reply = client.ping(2.0)
            if reply.get("error"):
                kill_tree(proc)
                raise ClientError(reply["error"])
            if reply.get("ready"):
                return proc, port
            if on_wait:
                on_wait(True)
        except ClientError as e:
            if "接続" not in str(e) and "応答" not in str(e):
                raise
            if on_wait:
                on_wait(False)
        time.sleep(0.5)
