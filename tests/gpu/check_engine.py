"""RTX 5080 で MotionBricks の SOMA エンジンが動くかの確認。サンプル動きは使わない。"""
import time

import numpy as np
import warp as wp

from motionbricks.io.usd import load_skeleton_from_usd
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH
from motionbricks.model.soma.engine import MotionBricksSOMAEngine
from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter

wp.init()
wp.set_device("cuda:0")
skel = load_skeleton_from_usd(get_asset_file("human", "soma_base_skel_minimal_no_root.usd"), SOMA_SKELETON_PRIM_PATH)
print("joints", skel.num_joints, skel.joint_names)
to_rt = SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())
ref = skel.reference_local_transforms.numpy().copy()
print("ref root (model)", ref[0])
root_rt = wp.mul(to_rt, wp.transform(*ref[0]))
ref[0] = np.array(root_rt)
print("ref root (runtime)", ref[0])
t0 = time.time()
with MotionBricksSOMAEngine(SOMA_N1_FP32_ONNX_PATH, model_to_runtime=to_rt) as engine, wp.ScopedStream(engine.stream):
    print("engine ready in %.1fs" % (time.time() - t0))
    ctx, tgt, out = engine.create_input(skel), engine.create_input(skel), engine.create_output(skel)
    c = np.repeat(ref[None], 4, axis=0)
    for off in ((1.0, 0.0), (0.0, 1.0), (0.0, -1.5)):
        t = ref.copy()
        t[0, 0] += off[0]
        t[0, 1] += off[1]
        ctx.update_from_poses(wp.array(c, dtype=wp.transform), stream=engine.stream)
        tgt.update_from_poses(wp.array(t[None], dtype=wp.transform), stream=engine.stream)
        t1 = time.time()
        engine.generate_animation(ctx, tgt, out)
        engine.synchronize()
        n = out.num_frames
        poses = out.poses[:n].numpy()
        print("target", off, "frames", n, "%.0f ms" % ((time.time() - t1) * 1000),
              "root start", poses[0, 0, :3].round(3), "end", poses[-1, 0, :3].round(3), "nan", bool(np.isnan(poses).any()))
        assert 18 <= n <= 58
print("CHECK OK")
