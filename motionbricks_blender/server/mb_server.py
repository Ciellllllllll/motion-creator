"""MotionBricks の推論サーバー。MotionBricks の uv 環境（Python 3.12）で動く。

127.0.0.1 だけで待ち受け、1 接続 1 リクエストで長さ付き JSON をやり取りする。
MotionBricks 同梱のサンプル動き（assets/motions）は一切読み込まない。
クリップはアドオン同梱のポーズ集フォルダにあるファイル名だけを受け付ける。
"""

import argparse
import ctypes
import math
import os
import queue
import socket
import sys
import threading
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import protocol  # noqa: E402

POSES_PER_TOKEN = 4
SAMPLE_RATE = 30.0


class Refuse(Exception):
    """利用者に返す日本語のエラー。"""


class State:
    ready = False
    error = ""
    version = ""


def watch_parent(pid: int) -> None:
    """親（Blender）が終了したら自分も終了する。"""
    if not pid or os.name != "nt":
        return
    handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        os._exit(0)

    def loop():
        ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        os._exit(0)

    threading.Thread(target=loop, daemon=True).start()


class Engine:
    """GPU を触る処理はすべてメインスレッドのこのクラスで行う。"""

    def __init__(self, library: Path, cache_dir: Path):
        import numpy as np
        import warp as wp

        import motionbricks
        from motionbricks.io.usd import load_skeleton_from_usd
        from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
        from motionbricks.model.config import MotionBricksInferenceConfig
        from motionbricks.model.soma.contract import SOMA_N1_CONTRACT, SOMA_N1_FP32_ONNX_PATH
        from motionbricks.model.soma.engine import MotionBricksSOMAEngine
        from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter

        self.np, self.wp = np, wp
        self.Config = MotionBricksInferenceConfig
        self.library = library.resolve()
        State.version = getattr(motionbricks, "__version__", "unknown")
        wp.config.log_level = wp.LOG_WARNING
        wp.init()
        wp.set_device("cuda:0")
        self.skel = load_skeleton_from_usd(
            get_asset_file("human", "soma_base_skel_minimal_no_root.usd"), SOMA_SKELETON_PRIM_PATH)
        self.names = list(self.skel.joint_names)
        self.parents = [int(p) for p in self.skel.parent_indices.numpy()]
        self.model_joints = list(getattr(SOMA_N1_CONTRACT, "joint_names", []))
        self.to_rt = SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())
        rest = self.skel.reference_local_transforms.numpy().copy()
        rest[0] = np.array(wp.mul(self.to_rt, wp.transform(*rest[0])))
        self.rest = rest
        self.i_left, self.i_right = self.names.index("LeftLeg"), self.names.index("RightLeg")
        self.engine = MotionBricksSOMAEngine(SOMA_N1_FP32_ONNX_PATH, model_to_runtime=self.to_rt,
                                             cache_dir=str(cache_dir), logger_severity="WARNING")
        self.stream = self.engine.stream
        with wp.ScopedStream(self.stream):
            self.context = self.engine.create_input(self.skel)
            self.target = self.engine.create_input(self.skel)
            self.output = self.engine.create_output(self.skel)
        self.clips = {}
        # 1 回生成して CUDA グラフまで準備しておく（最初の生成を待たせない）
        self.generate([self.rest.tolist()] * 4, [{"poses": [self.rest.tolist()], "use": 1, "early": 0}], None)

    # ---- 変換 ----
    def _poses(self, data, min_n=1, max_n=None):
        np = self.np
        try:
            arr = np.asarray(data, dtype=np.float32)
        except (ValueError, TypeError):
            raise Refuse("ポーズのデータが正しくありません")
        if arr.ndim != 3 or arr.shape[1:] != (len(self.names), 7) or arr.shape[0] < min_n \
                or (max_n and arr.shape[0] > max_n) or not np.isfinite(arr).all():
            raise Refuse("ポーズのデータが正しくありません")
        return arr

    def _wp(self, arr):
        return self.wp.array(arr, dtype=self.wp.transform, device=self.engine.device)

    @staticmethod
    def _round(arr):
        return [[[round(float(v), 6) for v in joint] for joint in pose] for pose in arr]

    # ---- コマンド ----
    def skeleton(self):
        return {"joint_names": self.names, "parents": self.parents,
                "rest": self._round([self.rest])[0], "model_joints": self.model_joints}

    def fk(self, pose):
        wp = self.wp
        arr = self._poses([pose], 1, 1)[0]
        with wp.ScopedStream(self.stream):
            out = wp.empty(len(self.names), dtype=wp.transform, device=self.engine.device)
            self.skel.compute_world_transforms(self._wp(arr), out, stream=self.stream)
            wp.synchronize_stream(self.stream)
            return self._round([out.numpy()])[0]

    def _clip(self, name):
        from motionbricks.io.bvh import load_bvh

        if not isinstance(name, str) or Path(name).name != name or not name.lower().endswith(".bvh"):
            raise Refuse("ポーズ集にない動きです")
        path = (self.library / name).resolve()
        if path.parent != self.library or not path.is_file():
            raise Refuse("ポーズ集にない動きです")
        if name not in self.clips:
            self.clips[name] = load_bvh(str(path), self.skel)
        return self.clips[name]

    def _facing(self, root, left, right):
        """腰の左右の関節から、ワールドでの向き（Z 軸まわりの角度）を求める。"""
        wp = self.wp
        root_t = wp.transform(*root)
        lp = wp.transform_get_translation(wp.mul(root_t, wp.transform(*left)))
        rp = wp.transform_get_translation(wp.mul(root_t, wp.transform(*right)))
        rx, ry = rp[0] - lp[0], rp[1] - lp[1]
        return math.atan2(rx, -ry)   # forward = up × right = (-ry, rx)

    def sample(self, name, time_s, anchor, placement):
        wp, np = self.wp, self.np
        clip = self._clip(name)
        if anchor not in (0, 3):
            raise Refuse("anchor が正しくありません")
        try:
            x, y, yaw = (float(v) for v in placement)
            time_s = float(time_s)
        except (TypeError, ValueError):
            raise Refuse("配置が正しくありません")
        duration = max(0.0, (clip.num_frames - 1) / clip.sample_rate)
        with wp.ScopedStream(self.stream):
            poses = wp.empty((POSES_PER_TOKEN, len(self.names)), dtype=wp.transform, device=self.engine.device)

            def t_of(i):
                return min(max(time_s + (i - anchor) / SAMPLE_RATE, 0.0), duration)

            clip.sample(t_of(anchor), poses[anchor], root_offset=self.to_rt, stream=self.stream)
            wp.synchronize_stream(self.stream)
            a = poses.numpy()[anchor]
            turn = yaw - self._facing(a[0], a[self.i_left], a[self.i_right])
            q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), turn)
            to_origin = wp.transform(wp.vec3(-float(a[0][0]), -float(a[0][1]), 0.0), wp.quat_identity())
            place = wp.mul(wp.transform(wp.vec3(x, y, 0.0), q), to_origin)
            offset = wp.mul(place, self.to_rt)
            for i in range(POSES_PER_TOKEN):
                clip.sample(t_of(i), poses[i], root_offset=offset, stream=self.stream)
            wp.synchronize_stream(self.stream)
            out = poses.numpy()
        if not np.isfinite(out).all():
            raise Refuse("ポーズを取得できませんでした")
        return {"poses": self._round(out), "duration": duration}

    def generate(self, context, targets, token_counts):
        wp, np = self.wp, self.np
        ctx = self._poses(context, 4, 4)
        if not isinstance(targets, list) or not targets:
            raise Refuse("目標のポーズがありません")
        config = self.Config()
        if token_counts:
            try:
                config = self.Config(allowed_token_counts=tuple(int(c) for c in token_counts))
            except (ValueError, TypeError):
                raise Refuse("生成の長さの指定が正しくありません")
        frames, contacts, segments = [], [], []
        with wp.ScopedStream(self.stream):
            for tg in targets:
                try:
                    poses = self._poses(tg["poses"], 1, 4)
                    use = max(1, min(int(tg.get("use", 1)), len(poses)))
                    early = max(0, int(tg.get("early", 0)))
                except (KeyError, TypeError, ValueError, AttributeError):
                    raise Refuse("目標のポーズが正しくありません")
                self.context.update_from_poses(self._wp(ctx), stream=self.stream)
                self.target.update_from_poses(self._wp(poses[:use]), stream=self.stream)
                self.engine.generate_animation(self.context, self.target, self.output, config=config)
                self.engine.synchronize()
                n = int(self.output.num_frames)
                played = max(POSES_PER_TOKEN, n - early)
                out = self.output.poses[:played].numpy().copy()
                con = self.output.foot_contacts[:played].numpy().copy()
                if not np.isfinite(out).all():
                    raise Refuse("生成に失敗しました")
                skip = 1 if frames else 0   # 2 区間目以降の先頭は前の区間の最後と同じ
                start = len(frames)
                frames.extend(out[skip:])
                contacts.extend(con[skip:])
                segments.append([start, len(frames) - 1])
                ctx = out[played - POSES_PER_TOKEN:played]
        return {"frames": self._round(frames),
                "contacts": [[round(float(v), 4) for v in row] for row in contacts],
                "segments": segments}

    def close(self):
        try:
            self.engine.close()
        except Exception:
            pass


def serve(port: int, library: Path, cache_dir: Path) -> None:
    jobs = queue.Queue()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", port))
    listener.listen(8)

    def accept_loop():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            try:
                conn.settimeout(60)
                request = protocol.recv(conn)
            except Exception:
                conn.close()
                continue
            cmd = request.get("cmd") if isinstance(request, dict) else None
            if cmd == "ping" or not State.ready:
                reply = {"ok": True, "ready": State.ready, "version": State.version, "error": State.error}
                if cmd != "ping":
                    reply = {"ok": False, "error": State.error or "MotionBricksを準備中です"}
                try:
                    protocol.send(conn, reply)
                finally:
                    conn.close()
                continue
            jobs.put((conn, request))

    threading.Thread(target=accept_loop, daemon=True).start()
    try:
        engine = Engine(library, cache_dir)
    except Exception as e:
        traceback.print_exc()
        State.error = f"MotionBricksの準備に失敗しました: {type(e).__name__}"
        threading.Event().wait(30)   # エラーを ping で伝える時間
        return
    State.ready = True
    print("READY", flush=True)

    while True:
        conn, request = jobs.get()
        stop = False
        try:
            cmd = request.get("cmd")
            if cmd == "skeleton":
                reply = engine.skeleton()
            elif cmd == "fk":
                reply = {"world": engine.fk(request.get("pose"))}
            elif cmd == "sample":
                reply = engine.sample(request.get("clip"), request.get("time", 0.0),
                                      request.get("anchor", 0), request.get("placement", [0, 0, 0]))
            elif cmd == "generate":
                reply = engine.generate(request.get("context"), request.get("targets"),
                                        request.get("token_counts"))
            elif cmd == "shutdown":
                reply, stop = {}, True
            else:
                raise Refuse("不明な命令です")
            reply["ok"] = True
        except Refuse as e:
            reply = {"ok": False, "error": str(e)}
        except Exception:
            traceback.print_exc()
            reply = {"ok": False, "error": "MotionBricksの処理中にエラーが起きました"}
        try:
            protocol.send(conn, reply)
        except Exception:
            pass
        finally:
            conn.close()
        if stop:
            break
    listener.close()
    engine.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--library", default=str(HERE.parent / "library"))
    parser.add_argument("--cache-dir", default=str(Path.cwd() / ".trt-cache"))
    args = parser.parse_args()
    watch_parent(args.parent_pid)
    serve(args.port, Path(args.library), Path(args.cache_dir))
    os._exit(0)


if __name__ == "__main__":
    main()
