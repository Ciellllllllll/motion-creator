"""足の固定（接地中の足を滑らせない）。

MotionBricks が返す足の接地の確信度から接地区間を作り、Mixamo キャラクターへ
転送した動きの脚を 2 本骨の IK で直す。接地中は足（母指球）を区間の中央値の位置に
留め、足裏を床に着ける。足の向きと膝の曲がる向きは保つ。
"""

import math

import bpy
from mathutils import Matrix, Vector

from . import rig_transfer as mt

CONTACT_THRESHOLD = 0.5
MIN_FRAMES = 3
MAX_GAP = 2
BLEND_FRAMES = 4
REACH_LIMIT = 0.98


class RefineError(ValueError):
    def __init__(self, stage, message):
        super().__init__(message)


class Feet:
    """Sole reference points of a Mixamo rig's feet.

    The rest floor is the lowest foot-chain joint along the rig's own "up"
    (the rest pose need not be upright in world space); heel / ball / toe-tip
    sole points sit directly below the ankle / ball / toe tip on that floor and
    move rigidly with the foot (heel, ball) or toe (tip) bone.
    """

    def __init__(self, arm):
        prefix = mt.detect_mixamo_prefix(arm)
        if prefix is None:
            raise RefineError("Input check", f"'{arm.name}' is not a Mixamo rig.")
        self.arm = arm
        self.prefix = prefix
        bones = arm.data.bones
        mw = arm.matrix_world
        try:
            self.frame = mt.RestFrame(arm, mt.mixamo_marks(prefix))
        except KeyError as e:
            raise RefineError("Input check", f"Missing landmark bone {e}")
        self.sides = {}
        for side, word in (("L", "Left"), ("R", "Right")):
            names = {k: prefix + word + k for k in ("UpLeg", "Leg", "Foot", "ToeBase")}
            missing = [n for n in names.values() if n not in bones]
            if missing:
                raise RefineError("Input check", "Missing leg bones: " + ", ".join(missing))
            end = bones.get(prefix + word + "Toe_End")
            toe = bones[names["ToeBase"]]
            ankle, ball = mw @ bones[names["Foot"]].head_local, mw @ toe.head_local
            if end is not None:
                tip = mw @ end.head_local
            else:  # bone tails are guesses in glTF imports: extend the foot instead
                along = self.frame.flatten(ball) - self.frame.flatten(ankle)
                tip = ball + along.normalized() * (0.6 * along.length) if along.length > 1e-6 else ball
            self.sides[side] = {"names": names, "rest": {"ankle": ankle, "ball": ball, "tip": tip}}
        self.rest_floor = self.frame.floor
        for side, d in self.sides.items():
            foot_rest = mw @ bones[d["names"]["Foot"]].matrix_local
            toe_rest = mw @ bones[d["names"]["ToeBase"]].matrix_local
            flat = {k: self.frame.flatten(p) for k, p in d["rest"].items()}
            d["local"] = {
                "heel": foot_rest.inverted() @ flat["ankle"],
                "ball": foot_rest.inverted() @ flat["ball"],
                "tip": toe_rest.inverted() @ flat["tip"],
            }
            ul, lg, ft = (bones[d["names"][k]] for k in ("UpLeg", "Leg", "Foot"))
            d["l1"] = ((mw @ lg.head_local) - (mw @ ul.head_local)).length
            d["l2"] = ((mw @ ft.head_local) - (mw @ lg.head_local)).length

    def points(self, side, pose):
        """World sole points {'heel','ball','tip'} for an armature-space pose."""
        d = self.sides[side]
        mw = self.arm.matrix_world
        foot = mw @ pose[d["names"]["Foot"]]
        toe = mw @ pose[d["names"]["ToeBase"]]
        return {
            "heel": foot @ d["local"]["heel"],
            "ball": foot @ d["local"]["ball"],
            "tip": toe @ d["local"]["tip"],
        }

    def joint(self, side, key, pose):
        return self.arm.matrix_world @ pose[self.sides[side]["names"][key]].translation


def _runs(flags, frames):
    out, start = [], None
    for f, on in zip(frames, flags):
        if on and start is None:
            start = f
        if not on and start is not None:
            out.append((start, prev))
            start = None
        prev = f
    if start is not None:
        out.append((start, frames[-1]))
    return out


def clean_intervals(intervals, min_frames, max_gap):
    merged = []
    for s, e in sorted(intervals):
        if merged and s - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s + 1 >= min_frames]



def _smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def contact_weight(frame, intervals, blend):
    """(weight, interval) — 1 inside an interval, easing to 0 over *blend* frames."""
    best = (0.0, None)
    for s, e in intervals:
        if s <= frame <= e:
            return 1.0, (s, e)
        dist = s - frame if frame < s else frame - e
        if blend > 0 and dist <= blend:
            w = _smoothstep(1.0 - dist / (blend + 1))
            if w > best[0]:
                best = (w, (s, e))
    return best


def two_bone_ik(H, K, A, T, l1, l2, reach_limit, fallback_pole):
    """New knee position and the reachable ankle target. Returns (K', T', clamped)."""
    d_vec = T - H
    d = d_vec.length
    if d < 1e-9:
        return K.copy(), A.copy(), True
    u = d_vec / d
    # Never ask for more extension than max(limit, what the original pose already has).
    original = min((A - H).length, (l1 + l2) * 0.9999)
    max_d = max((l1 + l2) * reach_limit, original)
    min_d = abs(l1 - l2) + 1e-6
    clamped = d > max_d + 1e-7 or d < min_d
    d = min(max(d, min_d), max_d)
    T_used = H + u * d
    pole = (K - H) - u * (K - H).dot(u)
    if pole.length < 1e-6:
        pole = fallback_pole - u * fallback_pole.dot(u)
    pole.normalize()
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    return H + u * a + pole * h, T_used, clamped


def apply_foot_lock(target, frames, bases_by_frame, contacts, floor, blend, reach_limit,
                    lock=True, fix_penetration=True):
    """Adjust leg bases in *bases_by_frame* (modified in place).

    Contact frames move the foot so its ball stays at the interval's anchor and
    its sole rests on the floor; outside contacts only floor penetration is
    lifted.  Foot orientation is preserved; the knee keeps its bend plane.
    Returns (changed_bones, warnings).
    """
    feet = Feet(target)
    rest_rel = mt.rest_relative(target)
    order = mt.bones_parent_first(target)
    mw = target.matrix_world
    q_obj = mw.to_3x3().normalized().to_quaternion()
    q_obj_inv = q_obj.inverted()
    poses = {f: mt.pose_from_bases(target, bases_by_frame[f], rest_rel, order) for f in frames}

    anchors = {}
    for side in ("L", "R"):
        for s, e in contacts.get(side, []):
            xs, ys = [], []
            for f in range(s, e + 1):
                if f in poses:
                    p = feet.points(side, poses[f])["ball"]
                    xs.append(p.x)
                    ys.append(p.y)
            if xs:
                xs.sort()
                ys.sort()
                anchors[(side, s, e)] = (xs[len(xs) // 2], ys[len(ys) // 2])

    clamped = {"L": [], "R": []}
    flipped = {"L": [], "R": []}
    changed = set()
    for f in frames:
        pose = poses[f]
        for side in ("L", "R"):
            names = feet.sides[side]["names"]
            pts = feet.points(side, pose)
            w, interval = contact_weight(f, contacts.get(side, []), blend) if lock else (0.0, None)
            delta = Vector((0, 0, 0))
            lowest = min(p.z for p in pts.values())
            if w > 0 and interval is not None and (side, *interval) in anchors:
                ax, ay = anchors[(side, *interval)]
                delta.x = w * (ax - pts["ball"].x)
                delta.y = w * (ay - pts["ball"].y)
                touch = floor - lowest          # lowest sole point rests on the floor
            else:
                touch = 0.0
            under = max(0.0, floor - lowest) if fix_penetration else 0.0
            delta.z = w * touch + (1 - w) * under
            if delta.length < 1e-7:
                continue

            H = feet.joint(side, "UpLeg", pose)
            K = feet.joint(side, "Leg", pose)
            A = feet.joint(side, "Foot", pose)
            l1, l2 = (K - H).length, (A - K).length
            # knees bend forward: facing from the posed hips (rest may not be upright)
            fallback = (feet.joint("L", "UpLeg", pose) - feet.joint("R", "UpLeg", pose)).cross(Vector((0, 0, 1)))
            fallback = fallback.normalized() if fallback.length > 1e-9 else Vector((0, -1, 0))
            K2, T2, was_clamped = two_bone_ik(H, K, A, A + delta, l1, l2, reach_limit, fallback)
            if was_clamped:
                clamped[side].append(f)
            n0 = (K - H).cross(A - K)
            n1 = (K2 - H).cross(T2 - K2)
            if n0.length > 1e-9 and n1.length > 1e-9 and n0.dot(n1) < 0:
                flipped[side].append(f)

            r1 = (K - H).rotation_difference(K2 - H)
            r2 = (r1 @ (A - K)).rotation_difference(T2 - K2)
            thigh_w = r1 @ mt._rot(mw @ pose[names["UpLeg"]])
            shin_w = r2 @ r1 @ mt._rot(mw @ pose[names["Leg"]])
            foot_w = mt._rot(mw @ pose[names["Foot"]])

            bases = bases_by_frame[f]
            chain = ((names["UpLeg"], thigh_w), (names["Leg"], shin_w), (names["Foot"], foot_w))
            for name, q_world in chain:
                bone = target.data.bones[name]
                parent = pose[bone.parent.name] if bone.parent else Matrix.Identity(4)
                rest_m = parent @ rest_rel[name]
                desired = Matrix.LocRotScale(rest_m.translation, q_obj_inv @ q_world, Vector((1, 1, 1)))
                basis = rest_m.inverted() @ desired
                old = bases.get(name, Matrix.Identity(4))
                bases[name] = Matrix.LocRotScale(old.translation, basis.to_quaternion(), old.to_scale())
                pose[name] = parent @ rest_rel[name] @ bases[name]
                changed.add(name)
            # bones below the foot (toes) follow the moved foot
            for b in target.data.bones[names["Foot"]].children_recursive:
                pose[b.name] = pose[b.parent.name] @ rest_rel[b.name] @ bases.get(b.name, Matrix.Identity(4))

    warnings = []
    for side in ("L", "R"):
        if clamped[side]:
            warnings.append(f"{'Left' if side == 'L' else 'Right'} leg could not reach the foot target on "
                            f"{len(clamped[side])} frame(s) (e.g. {_frame_list(clamped[side])}); "
                            "the correction was limited there.")
        if flipped[side]:
            warnings.append(f"{'Left' if side == 'L' else 'Right'} knee flipped its bend direction on "
                            f"{len(flipped[side])} frame(s) (e.g. {_frame_list(flipped[side])}).")
    return changed, warnings


def _frame_list(frames, limit=6):
    shown = ", ".join(str(f) for f in frames[:limit])
    return shown + (" …" if len(frames) > limit else "")



def contact_intervals(contacts, start=1):
    """[フレーム][左かかと, 左つま先, 右かかと, 右つま先] の確信度 → {'L': [[s, e]], 'R': [...]}。"""
    frames = list(range(start, start + len(contacts)))
    out = {}
    for side, cols in (("L", (0, 1)), ("R", (2, 3))):
        flags = [max(row[c] for c in cols) > CONTACT_THRESHOLD for row in contacts]
        out[side] = [list(i) for i in clean_intervals(_runs(flags, frames), MIN_FRAMES, MAX_GAP)]
    return out


MAX_HIP_DROP = 0.04


def _smooth_max(values, hold, sigma):
    """必要量の列を、前後に少し広げてからなめらかにする（急な上下動を避ける）。"""
    n = len(values)
    held = [max(values[max(0, i - hold):i + hold + 1]) for i in range(n)]
    radius = int(3 * sigma)
    weights = [math.exp(-0.5 * (k / sigma) ** 2) for k in range(-radius, radius + 1)]
    total = sum(weights)
    return [sum(w * held[min(max(i + k, 0), n - 1)] for w, k in zip(weights, range(-radius, radius + 1))) / total
            for i in range(n)]


def lower_hips_to_reach(target, frames, bases, intervals, floor):
    """接地中の足が床に届かないフレームで、腰を少し下げる（最大 MAX_HIP_DROP）。腰のボーン名を返す。"""
    feet = Feet(target)
    rest_rel = mt.rest_relative(target)
    order = mt.bones_parent_first(target)
    mw = target.matrix_world
    hips = feet.prefix + "Hips"
    need = []
    for f in frames:
        pose = mt.pose_from_bases(target, bases[f], rest_rel, order)
        drop = 0.0
        for side in ("L", "R"):
            w, _ = contact_weight(f, intervals.get(side, []), BLEND_FRAMES)
            if w <= 0:
                continue
            low = min(p.z for p in feet.points(side, pose).values())
            H, K, A = (feet.joint(side, k, pose) for k in ("UpLeg", "Leg", "Foot"))
            goal = A + Vector((0, 0, w * (floor - low)))
            limit = ((K - H).length + (A - K).length) * REACH_LIMIT
            flat = (goal.xy - H.xy).length
            if limit > flat:
                drop = max(drop, H.z - (goal.z + math.sqrt(limit * limit - flat * flat)))
        need.append(max(0.0, drop))
    if max(need, default=0.0) < 1e-4:
        return None
    drops = [min(MAX_HIP_DROP, d) for d in _smooth_max(need, 3, 3.0)]
    bone = target.data.bones[hips]
    parent_rest = bone.parent.matrix_local if bone.parent else Matrix.Identity(4)
    to_basis = (parent_rest.inverted() @ bone.matrix_local).to_3x3().inverted() @ parent_rest.to_3x3().inverted()
    to_arm = mw.to_3x3().inverted()
    for f, d in zip(frames, drops):
        if d > 1e-5:
            basis = bases[f][hips].copy()
            basis.translation += to_basis @ (to_arm @ Vector((0, 0, -d)))
            bases[f][hips] = basis
    return hips


def lock_action(target, action, contacts, start=1, floor=0.0):
    """*target*（Mixamo キャラクター）に割り当てた *action* の脚を直す。警告の一覧を返す。"""
    intervals = contact_intervals(contacts, start)
    if not intervals["L"] and not intervals["R"]:
        return []
    frames = list(range(start, start + len(contacts)))
    sampler = mt.PoseSampler(target, action)
    bases = {f: sampler.bases(f) for f in frames}
    hips = lower_hips_to_reach(target, frames, bases, intervals, floor)
    changed, warnings = apply_foot_lock(target, frames, bases, intervals, floor, BLEND_FRAMES, REACH_LIMIT)
    if hips:
        changed.add(hips)
    order = [b.name for b in mt.bones_parent_first(target) if b.name in changed]
    if order:
        mt.write_bases(target, action, frames, [bases[f] for f in frames], order,
                       root_bones=(hips,) if hips else ())
    return warnings
