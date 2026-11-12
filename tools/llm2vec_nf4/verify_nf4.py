"""保存済み NF4 版を新しいプロセスで再読み込みし、参照経路（FP32）と文章特徴量を比較する（仕様 8.2 後半）。

実行: python -s verify_nf4.py
Kimodo 実行時と同じ LocalNF4Encoder（ハッシュ照合つき）で読み込む。
"""
import json

import common
from common import BASE_DIR, MODEL_DIR, VERIFY_DIR, WORK, load_json, write_json

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from local_nf4_encoder import LocalNF4Encoder


def structure_check():
    out = {"tensors": 0, "dtypes": {}, "nonfinite": [], "lora_or_adapter_keys": []}
    for f in sorted(MODEL_DIR.glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                t = h.get_tensor(k)
                out["tensors"] += 1
                out["dtypes"][str(t.dtype)] = out["dtypes"].get(str(t.dtype), 0) + 1
                if "lora" in k or "adapter" in k:
                    out["lora_or_adapter_keys"].append(k)
                if t.is_floating_point() and not torch.isfinite(t).all():
                    out["nonfinite"].append(k)
    out["pass"] = not out["nonfinite"] and not out["lora_or_adapter_keys"] and out["tensors"] > 0
    return out


def main():
    tol = load_json(common.TOOLS / "tolerances.json")["nf4"]
    ref = np.load(VERIFY_DIR / "ref_fp32.npz")
    texts = [str(t) for t in ref["texts"]]
    report = {"tolerance": tol, "prompts": len(texts), "structure": structure_check()}

    tb = AutoTokenizer.from_pretrained(str(BASE_DIR))
    tn = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    report["tokenizer_identical_ids"] = all(tb(t)["input_ids"] == tn(t)["input_ids"] for t in texts)

    enc = LocalNF4Encoder(str(MODEL_DIR))
    report["cuda_allocated_gb_after_load"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    feats, lengths = enc(texts)
    x = feats[:, 0].float().cpu().numpy()
    single, single_len = enc(texts[0])
    report["kimodo_call_shapes"] = {"batch": list(feats.shape), "lengths": sorted(set(lengths)),
                                    "single": list(single.shape), "single_length": single_len}
    report["repeat_max_abs_diff"] = float((single[0].float().cpu().numpy() - x[0]).__abs__().max())

    r = ref["reference"]
    cos = (r * x).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(x, axis=1))
    report["shape_matches_reference"] = list(x.shape) == list(r.shape) and x.shape[1] == tol["shape"][0]
    report["finite"] = bool(np.isfinite(x).all())
    report["cos_mean"] = float(cos.mean())
    report["cos_min"] = float(cos.min())
    report["per_prompt"] = [{"text": t, "cos": float(c)} for t, c in zip(texts, cos)]
    # 参考: 統合済み FP16 と参照の差（NF4 化による劣化の切り分け用）
    m = ref["merged"]
    report["merged_fp16_cos_min_vs_reference"] = float(
        ((r * m).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(m, axis=1))).min())
    report["pass"] = bool(report["structure"]["pass"] and report["tokenizer_identical_ids"]
                          and report["shape_matches_reference"] and report["finite"]
                          and report["cos_mean"] >= tol["cosine_mean_ge"] and report["cos_min"] >= tol["cosine_each_ge"])
    np.savez(VERIFY_DIR / "nf4.npz", texts=np.array(texts), nf4=x)
    write_json(WORK / "logs" / "verify_nf4.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "per_prompt"}, ensure_ascii=False, indent=1))
    for p in sorted(report["per_prompt"], key=lambda p: p["cos"])[:5]:
        print(f"  {p['cos']:.5f}  {p['text']}")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
