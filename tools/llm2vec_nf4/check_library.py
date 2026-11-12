"""再生成したポーズ集を機械検査し、既存ポーズ集と比べる（仕様 8.3「フレーム数・骨格・数値・ファイル形式」）。

実行: python -s check_library.py <新ポーズ集フォルダー> [<既存ポーズ集フォルダー>] [--out report.json]

検査（合否）:
- ファイル形式: BVH として読め、index.json の全項目のファイルがある
- 骨格: 関節名・親子・OFFSET・CHANNELS が既存ポーズ集と完全一致
- フレーム数・フレーム時間: 既存と一致（尺 × 30fps）
- 数値: 全値が有限、各行の列数がチャンネル数と一致
- 動作の意味の簡易判定（下表）: 既存ポーズ集で成り立つ判定は、新ポーズ集でも成り立つこと
記録のみ: 既存との差（腰の軌跡・関節角の差）
"""
import json
import math
import sys
from pathlib import Path

import bvhio
import numpy as np

CM = 0.01


def parse_text(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    mi = lines.index("MOTION")
    hierarchy = [l.strip() for l in lines[:mi]]
    frames = int(lines[mi + 1].split()[1])
    frame_time = float(lines[mi + 2].split()[2])
    rows = [l.split() for l in lines[mi + 3:] if l.strip()]
    nch = sum(int(l.split()[1]) for l in hierarchy if l.startswith("CHANNELS"))
    data = np.array([[float(v) for v in r] for r in rows]) if all(len(r) == nch for r in rows) else None
    return {"hierarchy": hierarchy, "frames": frames, "frame_time": frame_time, "rows": len(rows),
            "channels": nch, "data": data}


def world_tracks(path: Path, names=("Hips", "Head", "LeftHand", "RightHand", "LeftShoulder", "RightShoulder",
                                    "LeftLeg", "RightLeg", "LeftFoot", "RightFoot")):
    root = bvhio.readAsHierarchy(str(path))
    joints = {j.Name: j for j, _, _ in root.layout()}
    n = len(root.Keyframes)
    out = {k: np.zeros((n, 3)) for k in names}
    for f in range(n):
        root.loadPose(f)
        for k in names:
            p = joints[k].PositionWorld
            out[k][f] = (p.x * CM, p.y * CM, p.z * CM)  # Y が上
    return out


def facing(tr, f):
    d = tr["RightLeg"][f] - tr["LeftLeg"][f]
    return math.atan2(d[2], d[0])


def semantics(cid, tr, stand_h):
    hips = tr["Hips"]
    horiz = np.linalg.norm(hips[-1, [0, 2]] - hips[0, [0, 2]])
    dur = len(hips) / 30.0
    speed = horiz / dur
    s = {"hips_start_m": round(float(hips[0, 1]), 3), "hips_min_m": round(float(hips[:, 1].min()), 3),
         "hips_max_m": round(float(hips[:, 1].max()), 3), "horizontal_m": round(float(horiz), 3),
         "speed_mps": round(float(speed), 3)}
    yaw = math.degrees(abs(math.atan2(math.sin(facing(tr, -1) - facing(tr, 0)), math.cos(facing(tr, -1) - facing(tr, 0)))))
    s["yaw_change_deg"] = round(yaw, 1)
    rh = tr["RightHand"][:, 1] - tr["RightShoulder"][:, 1]
    lh = tr["LeftHand"][:, 1] - tr["LeftShoulder"][:, 1]
    s["right_hand_above_shoulder_max_m"] = round(float(rh.max()), 3)
    s["left_hand_above_shoulder_max_m"] = round(float(lh.max()), 3)
    foot_min = min(tr["LeftFoot"][:, 1].min(), tr["RightFoot"][:, 1].min())
    s["foot_min_height_m"] = round(float(foot_min), 3)
    checks = {
        "stand": {"stays_in_place": horiz < 0.3},
        "walk": {"moves_forward": 0.5 < speed < 2.2},
        "jog": {"moves_faster_than_walk_range": 1.2 < speed < 4.5},
        "run": {"moves_fast": speed > 2.0},
        "crouch": {"hips_low": hips[:, 1].min() < 0.8 * stand_h},
        "crouch_walk": {"hips_low": hips[:, 1].mean() < 0.85 * stand_h, "moves": horiz > 0.4},
        "jump": {"hips_rise": hips[:, 1].max() - hips[0, 1] > 0.15},
        "sit": {"hips_low_at_end": hips[-1, 1] < 0.8 * stand_h},
        "wave": {"right_hand_raised": rh.max() > 0.05, "right_not_left": rh.max() > lh.max() + 0.1},
        "turn": {"turns_around": yaw > 120},
    }[cid]
    checks["feet_not_below_floor"] = foot_min > -0.05
    return s, {k: bool(v) for k, v in checks.items()}


def main():
    args = sys.argv[1:]
    out_path = None
    if "--out" in args:
        i = args.index("--out")
        out_path = Path(args[i + 1])
        del args[i:i + 2]
    new = Path(args[0])
    old = Path(args[1]) if len(args) > 1 else None
    idx = json.loads((new / "index.json").read_text(encoding="utf-8"))["items"]
    clips = [i for i in idx if not i.get("static")]
    report = {"new": str(new), "old": str(old) if old else None, "clips": {}}
    ok = True
    stand_h = {}
    tracks = {}
    for lib in ([new] + ([old] if old else [])):
        tracks[lib] = {i["id"]: world_tracks(lib / i["file"]) for i in clips}
        stand_h[lib] = float(tracks[lib]["stand"]["Hips"][:, 1].mean())
    for item in clips:
        cid = item["id"]
        r = {"file_exists": (new / item["file"]).is_file()}
        tn = parse_text(new / item["file"])
        r.update(frames=tn["frames"], rows=tn["rows"], frame_time=tn["frame_time"],
                 rows_match_frames=tn["rows"] == tn["frames"], columns_ok=tn["data"] is not None,
                 finite=bool(tn["data"] is not None and np.isfinite(tn["data"]).all()),
                 frame_time_30fps=abs(tn["frame_time"] - 1 / 30) < 1e-9)
        r["semantics"], r["semantic_checks"] = semantics(cid, tracks[new][cid], stand_h[new])
        if old:
            to = parse_text(old / item["file"])
            r["skeleton_identical_to_old"] = tn["hierarchy"] == to["hierarchy"]
            r["frames_equal_old"] = tn["frames"] == to["frames"]
            so, co = semantics(cid, tracks[old][cid], stand_h[old])
            r["old_semantics"], r["old_semantic_checks"] = so, co
            r["semantic_regressions"] = [k for k, v in co.items() if v and not r["semantic_checks"][k]]
            a, b = tracks[new][cid]["Hips"], tracks[old][cid]["Hips"]
            r["hips_path_mean_diff_m"] = round(float(np.linalg.norm(a - b, axis=1).mean()), 3)
            r["channels_mean_abs_diff"] = round(float(np.abs(tn["data"] - to["data"]).mean()), 3)
        required = ["file_exists", "rows_match_frames", "columns_ok", "finite", "frame_time_30fps"]
        if old:
            required += ["skeleton_identical_to_old", "frames_equal_old"]
        r["pass"] = all(r[k] for k in required) and (not old or not r["semantic_regressions"])
        if not old:
            r["pass"] = r["pass"] and all(r["semantic_checks"].values())
        ok &= r["pass"]
        report["clips"][cid] = r
    report["all_pass"] = bool(ok)
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if out_path:
        out_path.write_text(text, encoding="utf-8")
    for cid, r in report["clips"].items():
        print(f"{cid:12s} pass={r['pass']} frames={r['frames']} sem={r['semantic_checks']}"
              + (f" old={r['old_semantic_checks']} reg={r['semantic_regressions']}" if old else ""))
    print("all_pass:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
