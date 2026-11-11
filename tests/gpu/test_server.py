import copy
import json
import math
import pathlib
import socket

import pytest

from motionbricks_blender import client, protocol

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
MODEL_JOINTS = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
                "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
                "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd",
                "RightHandMiddleEnd", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin",
                "RightFoot", "RightToeBase"]


def placed(rest, x, y):
    p = copy.deepcopy(rest)
    p[0][0], p[0][1] = x, y
    return p


def test_ping_ready(server):
    r = server.ping()
    assert r["ready"] is True and r["version"]


def test_skeleton(skel):
    assert len(skel["joint_names"]) == 77 == len(skel["parents"]) == len(skel["rest"])
    assert skel["model_joints"] == MODEL_JOINTS
    assert skel["parents"][0] == -1 and skel["joint_names"][0] == "Hips"
    assert 0.9 < skel["rest"][0][2] < 1.1          # Z 上：腰の高さ約 1 m
    assert all(len(j) == 7 for j in skel["rest"])


def test_rejects_sample_motions(server):
    for bad in ("../motionbricks/assets/motions/soma/locomotion/idle/idle_loop_001__A268.bvh",
                "C:/x/assets/motions/a.bvh", "sub/walk.bvh"):
        with pytest.raises(client.ClientError):
            server.sample(bad, 0.0, 0, [0, 0, 0])


def test_generate_reaches_target(server, skel):
    rest = skel["rest"]
    r = server.generate([rest] * 4, [{"poses": [placed(rest, 0.0, -1.5)], "use": 1, "early": 0}], None)
    frames = r["frames"]
    assert 18 <= len(frames) <= 58
    assert len(r["contacts"]) == len(frames) and len(r["contacts"][0]) == 4
    assert math.dist(frames[-1][0][:2], (0.0, -1.5)) < 0.10
    assert math.dist(frames[0][0][:2], (0.0, 0.0)) < 0.02
    assert r["segments"] == [[0, len(frames) - 1]]


def test_generate_chain_two_targets(server, skel):
    rest = skel["rest"]
    targets = [{"poses": [placed(rest, 0.0, -1.5)], "use": 1, "early": 0},
               {"poses": [placed(rest, 1.5, -1.5)], "use": 1, "early": 0}]
    r = server.generate([rest] * 4, targets, None)
    (a0, a1), (b0, b1) = r["segments"]
    assert a0 == 0 and b0 == a1 + 1 and b1 == len(r["frames"]) - 1
    assert math.dist(r["frames"][a1][0][:2], (0.0, -1.5)) < 0.10
    assert math.dist(r["frames"][-1][0][:2], (1.5, -1.5)) < 0.10
    # つなぎ目で腰が飛ばない
    assert math.dist(r["frames"][a1][0][:3], r["frames"][b0][0][:3]) < 0.08


def test_token_counts_change_length(server, skel):
    rest = skel["rest"]
    tg = [{"poses": [placed(rest, 0.0, -1.5)], "use": 1, "early": 0}]
    short = server.generate([rest] * 4, tg, [6])
    long = server.generate([rest] * 4, tg, [16])
    assert len(short["frames"]) == 18 and len(long["frames"]) == 58


def test_fk_matches_rest_height(server, skel):
    w = server.fk(skel["rest"])
    names = skel["joint_names"]
    assert w[names.index("Head")][2] > 1.5 and w[names.index("LeftToeEnd")][2] < 0.05
    (DATA / "fk_expected.json").write_text(json.dumps({"pose": skel["rest"], "world": w}), encoding="utf-8")


def test_server_survives_bad_request(server):
    with socket.create_connection(("127.0.0.1", server.port), timeout=10) as s:
        protocol.send(s, {"cmd": "generate", "context": "x"})
        r = protocol.recv(s)
    assert r["ok"] is False and r["error"]
    assert server.ping()["ready"] is True
    with pytest.raises(client.ClientError):
        server.generate([[1, 2, 3]], [], None)
    assert server.ping()["ready"] is True


def test_unknown_port_is_japanese_error():
    with pytest.raises(client.ClientError) as e:
        client.Client(1, timeout=1).ping()
    assert any(ord(ch) > 0x3000 for ch in str(e.value))
