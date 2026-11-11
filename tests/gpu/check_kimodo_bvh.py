"""Kimodo の SOMA BVH を MotionBricks の骨格で読めるかの確認。"""
import sys

import numpy as np
import warp as wp

from motionbricks.io.bvh import load_bvh
from motionbricks.io.usd import load_skeleton_from_usd
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file

wp.init()
wp.set_device("cuda:0")
skel = load_skeleton_from_usd(get_asset_file("human", "soma_base_skel_minimal_no_root.usd"), SOMA_SKELETON_PRIM_PATH)
path = sys.argv[1]
own = load_bvh(path)                 # ファイル自身の骨格
mb = load_bvh(path, skel)            # MotionBricks の骨格へ
print("file joints", own.skeleton.num_joints, "frames", own.num_frames, "rate", own.sample_rate)
missing = [n for n in skel.joint_names if n not in own.skeleton.joint_names]
print("missing in file:", missing)


def world(anim, sk, t):
    pose = wp.empty(sk.num_joints, dtype=wp.transform)
    anim.sample(t, pose)
    out = wp.empty(sk.num_joints, dtype=wp.transform)
    sk.compute_world_transforms(pose, out)
    return out.numpy()


worst = 0.0
for t in (0.0, 0.5, 1.0, 1.5):
    a, b = world(own, own.skeleton, t), world(mb, skel, t)
    for name in ("Hips", "Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot", "LeftToeBase"):
        pa = a[own.skeleton.joint_index(name), :3]
        pb = b[skel.joint_index(name), :3]
        worst = max(worst, float(np.linalg.norm(pa - pb)))
    if t == 0.0:
        for name in ("Hips", "Head", "LeftFoot", "RightFoot", "LeftHand"):
            print(name, b[skel.joint_index(name), :3].round(3))
print("worst joint position difference: %.2f cm" % (worst * 100))
ref_w = wp.empty(skel.num_joints, dtype=wp.transform)
skel.compute_world_transforms(skel.reference_local_transforms, ref_w)
r = ref_w.numpy()
for name in ("Hips", "Head", "LeftFoot", "LeftToeEnd", "LeftHand", "RightHand"):
    print("mb rest", name, r[skel.joint_index(name), :3].round(3))
