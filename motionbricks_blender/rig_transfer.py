"""
Kimodo Blender Bridge — Motion tools

In Place
    Removes the horizontal (world X/Y) travel of the motion-root bone (Hips)
    so the generated motion plays on the spot.  Height (world Z) and every
    rotation — including turning — are kept.  The original keyframe values are
    stored on the Action, so turning In Place off restores them exactly.

Mixamo transfer
    Bakes the Kimodo source motion onto a Mixamo-style rig (``mixamorig:Hips``
    …) as a brand-new Action.  The target's previous Action is kept (fake
    user).  Rotations are transferred as world-space deltas from each rig's
    rest pose, with a per-bone direction alignment so rigs whose rest pose is
    not an exact T-pose (e.g. A-pose) still line up.  Root motion is scaled by
    the hip-height ratio of the two rigs.
"""

import re

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector


# ---------------------------------------------------------------------------
# F-Curve access (legacy Actions and Blender 4.4+ slotted Actions)
# ---------------------------------------------------------------------------

def _fcurves_of(obj):
    """Return the F-Curve collection that animates *obj*, or None."""
    ad = obj.animation_data
    if not ad or not ad.action:
        return None
    action = ad.action
    slot = getattr(ad, "action_slot", None)
    if hasattr(action, "slots") and slot is not None:
        from bpy_extras import anim_utils
        cb = anim_utils.action_get_channelbag_for_slot(action, slot)
        return cb.fcurves if cb else None
    return getattr(action, "fcurves", None)


def action_slot(action):
    """The first slot of a slotted Action (Blender 4.4+), else None."""
    slots = getattr(action, "slots", None)
    return slots[0] if slots else None


def action_fcurves(action, slot=None):
    """F-Curve collection of *action* (for *slot*, default its first slot)."""
    if action is None:
        return None
    if hasattr(action, "slots"):
        slot = slot or action_slot(action)
        if slot is None:
            return None
        from bpy_extras import anim_utils
        cb = anim_utils.action_ensure_channelbag_for_slot(action, slot)
        return cb.fcurves
    return action.fcurves


def action_groups(action, slot=None):
    if hasattr(action, "slots"):
        slot = slot or action_slot(action)
        from bpy_extras import anim_utils
        return anim_utils.action_ensure_channelbag_for_slot(action, slot).groups
    return action.groups


def bind_action(obj, action, slot=None):
    """Assign *action* to *obj* and bind *slot* (default: the action's first slot).

    Re-assigning the slot that is already bound crashes Blender 5.0.1, so the
    slot is only set when it differs.
    """
    if obj.animation_data is None:
        obj.animation_data_create()
    ad = obj.animation_data
    ad.action = action
    if action is None or not hasattr(ad, "action_slot"):
        return
    slot = slot or action_slot(action)
    if slot is not None and ad.action_slot != slot:
        ad.action_slot = slot


def _new_action_for(obj, name):
    """Create an Action, bind it to *obj* and return (action, fcurves, groups)."""
    if obj.animation_data is None:
        obj.animation_data_create()
    ad = obj.animation_data
    action = bpy.data.actions.new(name)

    if hasattr(action, "slots") and hasattr(ad, "action_slot"):
        from bpy_extras import anim_utils
        slot = action.slots.new(id_type='OBJECT', name=obj.name)
        bind_action(obj, action, slot)
        cb = anim_utils.action_ensure_channelbag_for_slot(action, slot)
        return action, cb.fcurves, cb.groups

    ad.action = action
    return action, action.fcurves, action.groups


_LOC_PATH = re.compile(r'^pose\.bones\["(.+)"\]\.location$')


def _location_fcurves(obj):
    """{bone_name: {array_index: fcurve}} for every pose-bone location curve."""
    fcurves = _fcurves_of(obj)
    result = {}
    if not fcurves:
        return result
    for fc in fcurves:
        m = _LOC_PATH.match(fc.data_path)
        if m:
            result.setdefault(m.group(1), {})[fc.array_index] = fc
    return result


def _curve_varies(fc, eps=1e-6):
    pts = fc.keyframe_points
    if len(pts) < 2:
        return False
    vals = [kp.co[1] for kp in pts]
    return max(vals) - min(vals) > eps


# ---------------------------------------------------------------------------
# In Place
# ---------------------------------------------------------------------------

INPLACE_KEY = "mb_inplace_original"
_ROOT_NAMES = ("Hips", "hips", "Pelvis", "pelvis")


def find_motion_root(arm):
    """Return (bone_name, {idx: fcurve}) of the bone carrying the root travel."""
    loc = _location_fcurves(arm)
    for name in _ROOT_NAMES:
        if name in loc and name in arm.data.bones:
            return name, loc[name]
    # Fallback: the top-most bone whose location actually moves.
    for bone in arm.data.bones:
        curves = loc.get(bone.name)
        if curves and any(_curve_varies(fc) for fc in curves.values()):
            return bone.name, curves
    return None, None


def is_in_place(arm):
    ad = arm.animation_data if arm else None
    return bool(ad and ad.action and INPLACE_KEY in ad.action)


def _location_to_world_matrix(arm, bone):
    """3x3 matrix mapping the bone's pose-location channels to a world delta.

    Assumes the bone's parent is not animated (true for Kimodo's static Root).
    """
    if bone.use_local_location:
        local = bone.matrix_local.to_3x3()
    elif bone.parent:
        local = bone.parent.matrix_local.to_3x3()
    else:
        local = Matrix.Identity(3)
    return arm.matrix_world.to_3x3() @ local


def apply_in_place(arm):
    """Strip horizontal root travel. Returns (ok, message)."""
    if arm is None or arm.type != 'ARMATURE':
        return False, "No source armature."
    if is_in_place(arm):
        return True, "Already in place."

    bone_name, curves = find_motion_root(arm)
    if not bone_name:
        return False, f"'{arm.name}' has no root location animation."
    if sorted(curves) != [0, 1, 2]:
        return False, f"'{bone_name}' needs X/Y/Z location curves."

    to_world = _location_to_world_matrix(arm, arm.data.bones[bone_name])
    try:
        to_local = to_world.inverted()
    except ValueError:
        return False, "Armature transform is not invertible."

    fcs = [curves[0], curves[1], curves[2]]
    first = min(kp.co[0] for fc in fcs for kp in fc.keyframe_points)
    anchor = to_world @ Vector([fc.evaluate(first) for fc in fcs])

    # Compute every new value from the untouched curves before editing them.
    edits = []
    for idx, fc in enumerate(fcs):
        for kp in fc.keyframe_points:
            frame = kp.co[0]
            local = Vector([c.evaluate(frame) for c in fcs])
            world = to_world @ local
            world.x, world.y = anchor.x, anchor.y
            edits.append((kp, (to_local @ world)[idx] - local[idx]))

    original = {"bone": bone_name}
    for idx, fc in enumerate(fcs):
        flat = []
        for kp in fc.keyframe_points:
            flat += [kp.co[1], kp.handle_left[1], kp.handle_right[1]]
        original[str(idx)] = flat

    for kp, delta in edits:
        kp.co[1] += delta
        kp.handle_left[1] += delta
        kp.handle_right[1] += delta
    for fc in fcs:
        fc.update()

    arm.animation_data.action[INPLACE_KEY] = original
    return True, f"In Place: horizontal travel of '{bone_name}' removed."


def remove_in_place(arm):
    """Restore the original root travel. Returns (ok, message)."""
    if not is_in_place(arm):
        return True, "Not in place."
    action = arm.animation_data.action
    original = action[INPLACE_KEY].to_dict()
    curves = _location_fcurves(arm).get(original.get("bone"), {})

    for idx in range(3):
        fc = curves.get(idx)
        flat = original.get(str(idx), [])
        if fc is None or len(flat) != 3 * len(fc.keyframe_points):
            return False, ("Keyframes changed since In Place was applied — "
                           "cannot restore automatically.")

    for idx in range(3):
        fc = curves[idx]
        flat = original[str(idx)]
        for i, kp in enumerate(fc.keyframe_points):
            kp.co[1], kp.handle_left[1], kp.handle_right[1] = flat[3 * i:3 * i + 3]
        fc.update()

    del action[INPLACE_KEY]
    return True, "In Place off: original travel restored."


def sync_in_place(arm, enabled):
    """Make *arm* match the In Place checkbox. Returns (ok, message)."""
    if arm is None or arm.type != 'ARMATURE':
        return True, ""
    return apply_in_place(arm) if enabled else remove_in_place(arm)


# ---------------------------------------------------------------------------
# Mixamo transfer
# ---------------------------------------------------------------------------

_FINGERS = ("Index", "Middle", "Ring", "Pinky")


def _soma_to_mixamo_pairs():
    """[(kimodo_bone, mixamo_bone_without_prefix)] for the SOMA77 skeleton.

    Kimodo/SOMA names differ from Mixamo's: SOMA ``LeftLeg`` is the thigh
    (Mixamo ``LeftUpLeg``), ``LeftShin`` is Mixamo ``LeftLeg``, the three
    spine bones are shifted by one, and SOMA fingers have an extra metacarpal
    joint (``Index1``) so ``Index2..4`` map to Mixamo ``Index1..3``.
    """
    pairs = [
        ("Hips", "Hips"), ("Spine1", "Spine"), ("Spine2", "Spine1"),
        ("Chest", "Spine2"), ("Neck1", "Neck"), ("Head", "Head"),
    ]
    for side in ("Left", "Right"):
        pairs += [
            (f"{side}Shoulder", f"{side}Shoulder"),
            (f"{side}Arm", f"{side}Arm"),
            (f"{side}ForeArm", f"{side}ForeArm"),
            (f"{side}Hand", f"{side}Hand"),
            (f"{side}Leg", f"{side}UpLeg"),
            (f"{side}Shin", f"{side}Leg"),
            (f"{side}Foot", f"{side}Foot"),
            (f"{side}ToeBase", f"{side}ToeBase"),
        ]
        pairs += [(f"{side}HandThumb{i}", f"{side}HandThumb{i}") for i in (1, 2, 3)]
        for finger in _FINGERS:
            pairs += [(f"{side}Hand{finger}{i + 1}", f"{side}Hand{finger}{i}")
                      for i in (1, 2, 3)]
    return pairs


def _soma_aim_children():
    """{kimodo_bone: kimodo_child} used to measure each bone's rest direction."""
    aim = {
        "Spine1": "Spine2", "Spine2": "Chest", "Chest": "Neck1", "Neck1": "Head",
    }
    for side in ("Left", "Right"):
        aim.update({
            f"{side}Shoulder": f"{side}Arm",
            f"{side}Arm": f"{side}ForeArm",
            f"{side}ForeArm": f"{side}Hand",
            f"{side}Hand": f"{side}HandMiddle2",
            f"{side}Leg": f"{side}Shin",
            f"{side}Shin": f"{side}Foot",
            f"{side}Foot": f"{side}ToeBase",
            f"{side}HandThumb1": f"{side}HandThumb2",
            f"{side}HandThumb2": f"{side}HandThumb3",
        })
        for finger in _FINGERS:
            aim[f"{side}Hand{finger}2"] = f"{side}Hand{finger}3"
            aim[f"{side}Hand{finger}3"] = f"{side}Hand{finger}4"
    return aim


_MIXAMO_KEY_BONES = ("Hips", "Spine", "LeftUpLeg", "RightUpLeg", "LeftArm", "RightArm")


def detect_mixamo_prefix(arm):
    """Return the Mixamo bone-name prefix (e.g. ``'mixamorig:'``) or None."""
    names = set(arm.data.bones.keys())
    candidates = sorted({n[:-len("Hips")] for n in names if n.endswith("Hips")}, key=len)
    for prefix in candidates:
        if all(prefix + b in names for b in _MIXAMO_KEY_BONES):
            return prefix
    return None


def is_mixamo_rig(arm):
    return bool(arm and arm.type == 'ARMATURE' and detect_mixamo_prefix(arm) is not None)




# ---------------------------------------------------------------------------
# Pose evaluation straight from F-Curves (no scene frame changes)
# ---------------------------------------------------------------------------

_CHANNEL_PATH = re.compile(
    r'^pose\.bones\["(.+)"\]\.(location|rotation_quaternion|rotation_euler|'
    r'rotation_axis_angle|scale)$')


def _unescape(name):
    try:
        return bpy.utils.unescape_identifier(name)
    except AttributeError:
        return name.replace('\\"', '"')


def _rot(matrix):
    return matrix.to_3x3().normalized().to_quaternion()


def bones_parent_first(arm):
    order = []

    def visit(bone):
        order.append(bone)
        for child in bone.children:
            visit(child)

    for bone in arm.data.bones:
        if bone.parent is None:
            visit(bone)
    return order


def rest_relative(arm):
    """{bone: parent-rest⁻¹ · rest} for every bone (armature space)."""
    return {
        b.name: (b.parent.matrix_local.inverted() @ b.matrix_local) if b.parent else b.matrix_local.copy()
        for b in arm.data.bones
    }


class PoseSampler:
    """Evaluate an armature pose from an Action's F-Curves at any (sub)frame.

    Channels the Action does not animate fall back to the pose bone's current
    value, which is what Blender itself does.  Bones are assumed to inherit
    rotation and scale fully (the default, and what Mixamo / BVH rigs use).
    """

    def __init__(self, arm, action, slot=None):
        self.arm = arm
        self.action = action
        self.order = bones_parent_first(arm)
        self.rest_rel = rest_relative(arm)
        self.curves = {}
        fcurves = action_fcurves(action, slot) if action else None
        for fc in fcurves or []:
            m = _CHANNEL_PATH.match(fc.data_path)
            if m:
                name = _unescape(m.group(1))
                self.curves.setdefault(name, {}).setdefault(m.group(2), {})[fc.array_index] = fc
        self.defaults = {}
        for pb in arm.pose.bones:
            self.defaults[pb.name] = {
                "location": tuple(pb.location),
                "rotation_quaternion": tuple(pb.rotation_quaternion),
                "rotation_euler": tuple(pb.rotation_euler),
                "rotation_axis_angle": tuple(pb.rotation_axis_angle),
                "scale": tuple(pb.scale),
                "mode": pb.rotation_mode,
            }

    def animated(self, name):
        return name in self.curves

    def _channel(self, name, prop, size, frame):
        default = self.defaults[name][prop]
        curves = self.curves.get(name, {}).get(prop, {})
        return [curves[i].evaluate(frame) if i in curves else default[i] for i in range(size)]

    def basis(self, name, frame):
        mode = self.defaults[name]["mode"]
        loc = Vector(self._channel(name, "location", 3, frame))
        if mode == 'QUATERNION':
            q = Quaternion(self._channel(name, "rotation_quaternion", 4, frame))
            rot = q.normalized().to_matrix().to_4x4() if q.magnitude > 1e-12 else Matrix.Identity(4)
        elif mode == 'AXIS_ANGLE':
            a = self._channel(name, "rotation_axis_angle", 4, frame)
            axis = Vector(a[1:])
            rot = Matrix.Rotation(a[0], 4, axis) if axis.length > 1e-12 else Matrix.Identity(4)
        else:
            rot = Euler(self._channel(name, "rotation_euler", 3, frame), mode).to_matrix().to_4x4()
        scale = self._channel(name, "scale", 3, frame)
        return Matrix.Translation(loc) @ rot @ Matrix.Diagonal((scale[0], scale[1], scale[2], 1.0))

    def bases(self, frame):
        return {b.name: self.basis(b.name, frame) for b in self.order}

    def pose(self, frame, overrides=None):
        """{bone: armature-space pose matrix}; *overrides* replaces bases."""
        mats = {}
        for bone in self.order:
            name = bone.name
            basis = overrides[name] if overrides and name in overrides else self.basis(name, frame)
            parent = mats[bone.parent.name] if bone.parent else Matrix.Identity(4)
            mats[name] = parent @ self.rest_rel[name] @ basis
        return mats


def pose_from_bases(arm, bases, rest_rel=None, order=None):
    """FK: {bone: basis} → {bone: armature-space matrix} (missing = identity)."""
    rest_rel = rest_rel or rest_relative(arm)
    mats = {}
    for bone in order or bones_parent_first(arm):
        parent = mats[bone.parent.name] if bone.parent else Matrix.Identity(4)
        mats[bone.name] = parent @ rest_rel[bone.name] @ bases.get(bone.name, Matrix.Identity(4))
    return mats


# ---------------------------------------------------------------------------
# Rig-to-rig transfer (world-space rotation deltas + rest direction alignment)
# ---------------------------------------------------------------------------

def _height(arm, bone_name):
    """Rest height of a bone head above the armature origin, in world units."""
    head = arm.matrix_world @ arm.data.bones[bone_name].head_local
    return head.z - arm.matrix_world.translation.z


class RestFrame:
    """A rig's own body frame at rest, measured from landmarks (world space).

    Rest poses are not always upright in world space: glTF files written by
    Motion2MixamoRig, for example, store the Y-up → Z-up conversion in the
    animation, so the rest skeleton lies on its back.  ``up`` runs hips → head,
    ``right`` runs right hip → left hip, ``fwd`` = right × up, and the floor is
    the lowest foot-chain joint along ``up``.

    marks: {'hips', 'head', 'left_hip', 'right_hip', 'feet': [bone, …]}
    """

    def __init__(self, arm, marks):
        bones = arm.data.bones
        mw = arm.matrix_world

        def p(name):
            return mw @ bones[name].head_local

        self.up = (p(marks["head"]) - p(marks["hips"])).normalized()
        right = p(marks["left_hip"]) - p(marks["right_hip"])
        right -= self.up * right.dot(self.up)
        self.right = right.normalized()
        self.fwd = self.right.cross(self.up).normalized()
        self.basis = Matrix((self.right, self.up, self.fwd)).transposed()   # columns
        feet = [p(n) for n in marks["feet"] if n in bones]
        self.floor = min(f.dot(self.up) for f in feet) if feet else 0.0
        hips = p(marks["hips"])
        self.height = hips.dot(self.up) - self.floor
        self.pivot = hips - self.up * self.height        # hips projected onto the floor
        # Root reference: the rest hips stood upright (up snapped to its nearest
        # world axis, then turned to +Z about the floor point).  Upright rigs keep
        # their rest hips; it is the same point whichever side of a transfer the
        # rig is on, so A → B → A round trips are exact.
        axis = max(range(3), key=lambda i: abs(self.up[i]))
        snapped = Vector((0, 0, 0))
        snapped[axis] = 1.0 if self.up[axis] > 0 else -1.0
        upright = snapped.rotation_difference(Vector((0, 0, 1)))
        self.ref = self.pivot + upright @ (hips - self.pivot)

    def flatten(self, point):
        """*point* moved along ``up`` onto the rest floor."""
        return point - self.up * (point.dot(self.up) - self.floor)


def soma_marks():
    feet = [f"{s}{k}" for s in ("Left", "Right") for k in ("Foot", "ToeBase", "ToeEnd")]
    return {"hips": "Hips", "head": "Head", "left_hip": "LeftLeg", "right_hip": "RightLeg", "feet": feet}


def mixamo_marks(prefix):
    feet = [f"{prefix}{s}{k}" for s in ("Left", "Right") for k in ("Foot", "ToeBase", "Toe_End")]
    return {"hips": prefix + "Hips", "head": prefix + "Head", "left_hip": prefix + "LeftUpLeg",
            "right_hip": prefix + "RightUpLeg", "feet": feet}


class RigTransfer:
    """Precomputed mapping from a source rig to a target rig.

    pairs : [(source_bone, target_bone)]
    aim   : {source_bone: source_child} — measures each bone's rest direction.
    Rotations are copied as world-space deltas from each rig's rest pose; each
    target bone is first swung so its rest direction matches the source's.
    Root (hips) travel is scaled by the hip-height ratio of the two rigs.

    With landmarks (*src_marks* / *tgt_marks*), the target's rest pose is first
    re-oriented so its body frame matches the source's (``self.G``).  This keeps
    rigs whose rest pose is not upright, or faces another way, correct; for two
    upright rigs facing the same way G is the identity.
    """

    def __init__(self, source, target, pairs, aim, src_root, tgt_root, align_fingers=True,
                 src_marks=None, tgt_marks=None):
        self.source, self.target = source, target
        src_bones, tgt_bones = source.data.bones, target.data.bones
        self.mapping = {t: s for s, t in pairs if s in src_bones and t in tgt_bones}
        if self.mapping.get(tgt_root) != src_root:
            raise ValueError(f"Root bones '{src_root}' → '{tgt_root}' are missing.")
        self.src_root, self.tgt_root = src_root, tgt_root
        self.warnings = []

        src_mw, tgt_mw = source.matrix_world, target.matrix_world
        self.src_mw, self.tgt_mw = src_mw.copy(), tgt_mw.copy()
        self.tgt_mw_inv = tgt_mw.inverted()
        self.q_tgt_obj_inv = _rot(tgt_mw).inverted()

        self.src_frame = self.tgt_frame = None
        self.G = Quaternion()
        if src_marks and tgt_marks:
            try:
                self.src_frame = RestFrame(source, src_marks)
                self.tgt_frame = RestFrame(target, tgt_marks)
                self.G = (self.src_frame.basis @ self.tgt_frame.basis.transposed()).to_quaternion()
            except (KeyError, ValueError, ZeroDivisionError):
                self.src_frame = self.tgt_frame = None
        G = self.G

        self.src_rest_q_inv = {s: _rot(src_mw @ src_bones[s].matrix_local).inverted()
                               for s in self.mapping.values()}
        self.tgt_rest_q = {t: G @ _rot(tgt_mw @ tgt_bones[t].matrix_local) for t in self.mapping}

        src_to_tgt = {s: t for t, s in self.mapping.items()}
        self.align = {}
        for tgt_name, src_name in self.mapping.items():
            child = aim.get(src_name)
            if child and child in src_bones and child in src_to_tgt:
                d_src = (src_mw @ src_bones[child].head_local) - (src_mw @ src_bones[src_name].head_local)
                d_tgt = G @ ((tgt_mw @ tgt_bones[src_to_tgt[child]].head_local)
                             - (tgt_mw @ tgt_bones[tgt_name].head_local))
                if d_src.length > 1e-6 and d_tgt.length > 1e-6:
                    self.align[tgt_name] = d_tgt.rotation_difference(d_src)
        self.order = bones_parent_first(target)
        if align_fingers:  # distal finger bones follow their parent's alignment
            for bone in self.order:
                if (bone.name in self.mapping and bone.name not in self.align and "Hand" in bone.name
                        and bone.parent and bone.parent.name in self.align):
                    self.align[bone.name] = self.align[bone.parent.name]

        # World offset added to the source root before mapping (used to undo the
        # origin shift applied to Kimodo constraints).
        self.src_root_offset = Vector((0, 0, 0))
        if self.tgt_frame is not None:
            self.src_rest_root = self.src_frame.ref
            self.tgt_rest_root = self.tgt_frame.ref
            h_src, h_tgt = self.src_frame.height, self.tgt_frame.height
        else:
            self.src_rest_root = src_mw @ src_bones[src_root].head_local
            self.tgt_rest_root = tgt_mw @ tgt_bones[tgt_root].head_local
            h_src, h_tgt = _height(source, src_root), _height(target, tgt_root)
        self.root_scale = h_tgt / h_src if h_src > 1e-6 and h_tgt > 1e-6 else 1.0
        self.rest_rel = rest_relative(target)
        for b in self.order:
            if b.name in self.mapping and (not b.use_inherit_rotation or b.inherit_scale != 'FULL'):
                self.warnings.append(f"'{b.name}' does not fully inherit its parent; result may drift.")

    @property
    def mapped_bones(self):
        return [b.name for b in self.order if b.name in self.mapping]

    def bases(self, src_pose):
        """Target bases {bone: Matrix} for one source pose (armature space)."""
        out, mats = {}, {}
        for bone in self.order:
            name = bone.name
            parent = mats.get(bone.parent.name) if bone.parent else Matrix.Identity(4)
            rest_m = parent @ self.rest_rel[name]
            if name not in self.mapping:
                mats[name] = rest_m
                continue
            src_name = self.mapping[name]
            src_world = self.src_mw @ src_pose[src_name]
            delta = _rot(src_world) @ self.src_rest_q_inv[src_name]
            q_arm = self.q_tgt_obj_inv @ (delta @ self.align.get(name, Quaternion()) @ self.tgt_rest_q[name])
            if name == self.tgt_root:
                world_loc = self.tgt_rest_root + (src_world.translation + self.src_root_offset
                                                  - self.src_rest_root) * self.root_scale
                loc = self.tgt_mw_inv @ world_loc
            else:
                loc = rest_m.translation
            desired = Matrix.LocRotScale(loc, q_arm, Vector((1, 1, 1)))
            basis = rest_m.inverted() @ desired
            if name != self.tgt_root:
                basis = basis.to_quaternion().to_matrix().to_4x4()
            out[name] = basis
            mats[name] = desired
        return out


def sample_transfer(transfer, source_action, frames, frame_map=None):
    """[{bone: basis}] for each target frame; *frame_map* maps to source frames."""
    sampler = PoseSampler(transfer.source, source_action)
    return [transfer.bases(sampler.pose(frame_map(f) if frame_map else f)) for f in frames]


# ---------------------------------------------------------------------------
# Writing per-frame bases into an Action
# ---------------------------------------------------------------------------

def _bone_path(name):
    return f'pose.bones["{bpy.utils.escape_identifier(name)}"]'


def _replace_curve(fcurves, data_path, index, frames, values, group=None):
    """Replace keys of one channel on [frames[0], frames[-1]] with per-frame LINEAR keys.

    Keys outside that range are kept exactly (value, handles, interpolation).
    """
    fc = fcurves.find(data_path, index=index)
    lo, hi = frames[0] - 0.5, frames[-1] + 0.5
    kept = []
    if fc is None:
        fc = fcurves.new(data_path, index=index)
        if group is not None:
            try:
                fc.group = group
            except Exception:
                pass
    else:
        for kp in fc.keyframe_points:
            if kp.co[0] < lo or kp.co[0] > hi:
                kept.append((tuple(kp.co), tuple(kp.handle_left), tuple(kp.handle_right),
                             kp.interpolation, kp.handle_left_type, kp.handle_right_type))
        fc.keyframe_points.clear()

    rows = [(k[0], k) for k in kept] + [((float(f), float(v)), None) for f, v in zip(frames, values)]
    rows.sort(key=lambda r: r[0][0])
    fc.keyframe_points.add(len(rows))
    co = []
    for (f, v), _ in rows:
        co += [f, v]
    fc.keyframe_points.foreach_set("co", co)
    for kp, (_, k) in zip(fc.keyframe_points, rows):
        if k is None:
            kp.interpolation = 'LINEAR'
        else:
            kp.interpolation = k[3]
            kp.handle_left_type, kp.handle_right_type = k[4], k[5]
    fc.update()
    for kp, (_, k) in zip(fc.keyframe_points, rows):
        if k is not None:
            kp.handle_left, kp.handle_right = k[1], k[2]
    return fc


def write_bases(target, action, frames, bases_per_frame, bones, root_bones=()):
    """Key the given bones' bases on *frames* into *action* (in place).

    Rotations are written in each pose bone's rotation mode, continuous with
    the existing keys just before the range (quaternion sign / Euler wrap).
    """
    fcurves = action_fcurves(action)
    groups = action_groups(action)
    before = PoseSampler(target, action)
    for name in bones:
        pb = target.pose.bones[name]
        group = groups.get(name) or groups.new(name)
        base = _bone_path(name)
        mats = [b[name] for b in bases_per_frame]

        if name in root_bones:
            for i in range(3):
                _replace_curve(fcurves, base + ".location", i, frames,
                               [m.translation[i] for m in mats], group)

        mode = pb.rotation_mode
        prev_basis = before.basis(name, frames[0] - 1)
        if mode == 'QUATERNION':
            prev, quats = prev_basis.to_quaternion(), []
            for m in mats:
                q = m.to_quaternion()
                if prev.dot(q) < 0:
                    q.negate()
                quats.append(q)
                prev = q
            for i in range(4):
                _replace_curve(fcurves, base + ".rotation_quaternion", i, frames,
                               [q[i] for q in quats], group)
        elif mode == 'AXIS_ANGLE':
            vals = []
            for m in mats:
                axis, angle = m.to_quaternion().to_axis_angle()
                vals.append((angle, axis.x, axis.y, axis.z))
            for i in range(4):
                _replace_curve(fcurves, base + ".rotation_axis_angle", i, frames,
                               [v[i] for v in vals], group)
        else:
            prev, eulers = prev_basis.to_euler(mode), []
            for m in mats:
                e = m.to_euler(mode, prev)
                eulers.append(e)
                prev = e
            for i in range(3):
                _replace_curve(fcurves, base + ".rotation_euler", i, frames,
                               [e[i] for e in eulers], group)


def unique_action_name(base):
    """'base_001', 'base_002', … — never an existing Action name."""
    i = 1
    while f"{base}_{i:03d}" in bpy.data.actions:
        i += 1
    return f"{base}_{i:03d}"


# ---------------------------------------------------------------------------
# Kimodo (SOMA) ↔ Mixamo
# ---------------------------------------------------------------------------

def mixamo_transfer(source, target):
    """RigTransfer Kimodo SOMA → Mixamo rig (prefix detected)."""
    prefix = detect_mixamo_prefix(target)
    if prefix is None:
        raise ValueError(f"'{target.name}' is not a Mixamo rig "
                         "(expected bones like 'mixamorig:Hips').")
    if "Hips" not in source.data.bones:
        raise ValueError("The source armature has no 'Hips' bone — "
                         "is it a Kimodo (SOMA) armature?")
    pairs = [(s, prefix + m) for s, m in _soma_to_mixamo_pairs()]
    return RigTransfer(source, target, pairs, _soma_aim_children(), "Hips", prefix + "Hips",
                       src_marks=soma_marks(), tgt_marks=mixamo_marks(prefix))


def soma_from_mixamo_transfer(mixamo, soma):
    """RigTransfer Mixamo rig → Kimodo SOMA armature (inverse of mixamo_transfer)."""
    prefix = detect_mixamo_prefix(mixamo)
    if prefix is None:
        raise ValueError(f"'{mixamo.name}' is not a Mixamo rig.")
    to_mix = {s: prefix + m for s, m in _soma_to_mixamo_pairs()}
    pairs = [(m, s) for s, m in to_mix.items()]
    aim = {to_mix[s]: to_mix[c] for s, c in _soma_aim_children().items()
           if s in to_mix and c in to_mix}
    return RigTransfer(mixamo, soma, pairs, aim, prefix + "Hips", "Hips",
                       src_marks=mixamo_marks(prefix), tgt_marks=soma_marks())


def soma_rest_warning(source):
    bones = source.data.bones
    if "Spine1" in bones and "Spine2" in bones:
        d = source.matrix_world.to_3x3() @ (bones["Spine2"].head_local - bones["Spine1"].head_local)
        if d.length > 1e-6 and d.normalized().z < 0.85:
            return ("Kimodo rest pose is not upright — enable "
                    "'Use Standard T-Pose' before generating for best results.")
    return None


def reset_unmapped_pose(arm, mapped):
    """Put bones outside *mapped* at rest (the transfer assumes identity bases)."""
    for pb in arm.pose.bones:
        if pb.name not in mapped:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.rotation_euler = (0, 0, 0)
            pb.scale = (1, 1, 1)


def transfer_to_mixamo(context, source, target, action_name="Kimodo_Mixamo"):
    """Bake *source* motion onto Mixamo rig *target* as a new Action.

    Returns (action, n_mapped_bones, warnings).  Raises ValueError on bad input.
    """
    if source is None or target is None:
        raise ValueError("Set both the Kimodo source and the Mixamo target armature.")
    if source == target:
        raise ValueError("Source and target must be different armatures.")
    if not (source.animation_data and source.animation_data.action):
        raise ValueError(f"'{source.name}' has no animation to transfer.")

    transfer = mixamo_transfer(source, target)
    warnings = list(transfer.warnings)
    if len(transfer.mapping) < 10:
        warnings.append(f"Only {len(transfer.mapping)} bones matched — check the rigs.")
    if any(c.name.startswith("KIMODO_") for pb in target.pose.bones for c in pb.constraints):
        warnings.append("Target still has Kimodo retarget constraints; "
                        "they override the transferred keys. Remove them.")
    w = soma_rest_warning(source)
    if w:
        warnings.append(w)

    src_action = source.animation_data.action
    f_start, f_end = src_action.frame_range
    frames = list(range(int(round(f_start)), int(round(f_end)) + 1))
    bases = sample_transfer(transfer, src_action, frames)

    ad = target.animation_data
    old_action = ad.action if ad else None
    if old_action is not None:
        old_action.use_fake_user = True   # keep the user's previous animation

    reset_unmapped_pose(target, transfer.mapping)
    action, _, _ = _new_action_for(target, action_name)
    action["mb_source"] = source.name
    write_bases(target, action, frames, bases, transfer.mapped_bones, root_bones=(transfer.tgt_root,))
    action.use_fake_user = True
    return action, len(transfer.mapping), warnings
