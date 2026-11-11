"""生成の連鎖の組み立て（bpy 非依存）。

最初のポーズを出発点にして、途中・最後のポーズを順に目標にする。1 回の生成で
届かないほど離れている区間には、歩きの姿勢の中間目標を自動で足す。
"""

import math

MAX_STEP_M = 2.0
# 「もう一度」で結果を変えるための、生成の長さ（1 区間のトークン数）の候補。
# 既定（モデルが 8〜14 から選ぶ）は長めを選びやすいので、長さを 1 つに固定して変える。
_VARIANTS = (12, 10, 16, 8, 14, 6)


def _anchor_xy(window, kind):
    pose = window[3] if kind == "START" else window[0]
    return pose[0][0], pose[0][1]


def token_counts(variant: int):
    if variant <= 0:
        return None
    return [_VARIANTS[(variant - 1) % len(_VARIANTS)]]


def build_request(windows, kinds, walk_sampler, variant: int) -> dict:
    """生成サーバーへの依頼を作る。

    windows     : 各ポーズのワールドでの 4 ポーズ
    kinds       : 'START' / 'MIDDLE' / 'END'
    walk_sampler: (x, y, 向き) → 中間目標に使う歩きの 4 ポーズ
    戻り値の pose_targets は、利用者のポーズに対応する目標の番号。
    """
    if len(windows) < 2 or len(windows) != len(kinds) or kinds[0] != "START" or kinds[-1] != "END":
        raise ValueError("最初と最後のポーズが必要です")
    targets, pose_targets = [], []
    prev = _anchor_xy(windows[0], "START")
    for window, kind in zip(windows[1:], kinds[1:]):
        here = _anchor_xy(window, kind)
        dx, dy = here[0] - prev[0], here[1] - prev[1]
        dist = math.hypot(dx, dy)
        if dist > MAX_STEP_M:
            steps = math.ceil(dist / MAX_STEP_M)
            yaw = math.atan2(dy, dx)
            for i in range(1, steps):
                f = i / steps
                poses = walk_sampler(prev[0] + dx * f, prev[1] + dy * f, yaw)
                targets.append({"poses": poses[:2], "use": 2, "early": 8})
        if kind == "END":
            targets.append({"poses": window[:1], "use": 1, "early": 0})
        else:
            targets.append({"poses": window, "use": 4, "early": 0})
        pose_targets.append(len(targets) - 1)
        prev = here
    return {"context": windows[0], "targets": targets, "token_counts": token_counts(variant),
            "pose_targets": pose_targets}
