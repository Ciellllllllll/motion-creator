import json
import math
import pathlib

LIB = pathlib.Path(__file__).resolve().parents[2] / "motionbricks_blender" / "library"
NAMES = ["基本姿勢", "立ち", "歩き", "小走り", "走り", "しゃがみ", "しゃがみ歩き", "ジャンプ", "座る", "手を振る", "振り返る"]


def items():
    return json.loads((LIB / "index.json").read_text(encoding="utf-8"))["items"]


def test_index_complete():
    it = items()
    assert [i["name"] for i in it] == NAMES
    assert it[0]["id"] == "base" and it[0]["static"] is True
    for i in it:
        assert (LIB / i["file"]).is_file()
        assert 0 <= i["default_time"] <= i["duration"]


def test_no_sample_motion_origin():
    for i in items():
        assert i["prompt"] and isinstance(i["seed"], int) and i["model"].startswith("Kimodo-SOMA")


def test_all_clips_load(server, skel):
    for i in items():
        poses = server.sample(i["file"], i["default_time"], 0, [0, 0, 0], static=i.get("static", False))
        assert len(poses) == 4 and len(poses[0]) == 77
        flat = [v for p in poses for j in p for v in j]
        assert all(math.isfinite(v) for v in flat)
        assert 0.3 < poses[0][0][2] < 1.2, i["name"]     # 腰の高さが妥当


def _facing(server, skel, pose):
    w = server.fk(pose)
    n = skel["joint_names"]
    l, r = w[n.index("LeftLeg")], w[n.index("RightLeg")]
    return math.atan2(r[0] - l[0], -(r[1] - l[1]))


def test_placement_and_facing(server, skel):
    walk = next(i for i in items() if i["id"] == "walk")
    for anchor in (0, 3):
        for yaw in (0.0, 1.2, -2.5):
            poses = server.sample(walk["file"], walk["default_time"], anchor, [1.0, 2.0, yaw])
            assert math.dist(poses[anchor][0][:2], (1.0, 2.0)) < 1e-3
            d = _facing(server, skel, poses[anchor]) - yaw
            assert abs(math.atan2(math.sin(d), math.cos(d))) < 1e-2
    # 歩きは 4 ポーズの間に進んでいる
    poses = server.sample(walk["file"], walk["default_time"], 0, [0, 0, 0])
    assert math.dist(poses[0][0][:2], poses[3][0][:2]) > 0.03


def test_static_base_has_no_motion(server):
    base = items()[0]
    poses = server.sample(base["file"], base["default_time"], 3, [0.5, 0.5, 0.3], static=True)
    assert poses[0] == poses[1] == poses[2] == poses[3]
