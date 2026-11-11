import math

from motionbricks_blender import chain


def window(x, y, n=77):
    pose = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for _ in range(n)]
    pose[0][0], pose[0][1], pose[0][2] = x, y, 1.0
    return [[list(j) for j in pose] for _ in range(4)]


def sampler(calls):
    def f(x, y, yaw):
        calls.append((x, y, yaw))
        return window(x, y)
    return f


def xy(target):
    return tuple(target["poses"][0][0][:2])


def test_no_intermediate_when_close():
    calls = []
    req = chain.build_request([window(0, 0), window(0, -2)], ["START", "END"], sampler(calls), 0)
    assert len(req["targets"]) == 1 and calls == []
    assert req["targets"][0]["use"] == 1 and req["targets"][0]["early"] == 0
    assert req["context"] == window(0, 0) and req["pose_targets"] == [0]
    assert req["token_counts"] is None


def test_intermediates_when_far():
    calls = []
    req = chain.build_request([window(0, 0), window(0, -5), window(3, -5)], ["START", "MIDDLE", "END"],
                              sampler(calls), 0)
    t = req["targets"]
    assert len(t) == 5 and req["pose_targets"] == [2, 4]
    assert [(g["use"], g["early"]) for g in t] == [(2, 8), (2, 8), (4, 0), (2, 8), (1, 0)]
    pts = [(0.0, 0.0)] + [xy(g) for g in t]
    assert all(math.dist(a, b) <= chain.MAX_STEP_M + 1e-6 for a, b in zip(pts, pts[1:]))
    assert math.dist(xy(t[0]), (0, -5 / 3)) < 1e-6 and math.dist(xy(t[3]), (1.5, -5)) < 1e-6
    # 中間の目標は進む方向を向く
    assert abs(calls[0][2] - (-math.pi / 2)) < 1e-6 and abs(calls[2][2] - 0.0) < 1e-6


def test_middle_uses_four_poses_and_end_stops():
    req = chain.build_request([window(0, 0), window(0, -1), window(0, -2)], ["START", "MIDDLE", "END"],
                              sampler([]), 0)
    assert [(g["use"], g["early"], len(g["poses"])) for g in req["targets"]] == [(4, 0, 4), (1, 0, 1)]


def test_variant_token_counts():
    w = [window(0, 0), window(0, -2)]
    k = ["START", "END"]
    assert chain.build_request(w, k, sampler([]), 0)["token_counts"] is None
    got = [chain.build_request(w, k, sampler([]), v)["token_counts"] for v in range(1, 8)]
    assert got == [[12], [10], [16], [8], [14], [6], [12]]


def test_needs_start_and_end():
    try:
        chain.build_request([window(0, 0)], ["START"], sampler([]), 0)
        assert False
    except ValueError:
        pass
