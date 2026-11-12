"""NF4 版が基準に届かない原因の切り分け（仕様 8.2「不合格ならトークナイズ、プーリング、統合順序、量子化設定を切り分ける」）。

実行: python -s diagnose_nf4.py
参照経路（verify/ref_fp32.npz の FP32）との コサイン類似度を、次の条件で測る。
  nf4_fp16      : 自作 NF4、非量子化部分 FP16（採用候補・Kimodo 実行時と同じ）
  nf4_bf16      : 自作 NF4、非量子化部分 BF16（Kimodo の既定プリセット dtype）
  fp16_compute  : 量子化なし・統合済み FP16 重みで FP16 計算（FP16 演算だけの誤差）
  third_party   : 既存の第三者 NF4（~/.kimodo-venv/llm2vec-model）。読み取りのみ・診断専用
"""
import gc
import json
import os
from pathlib import Path

import common
from common import MERGED_DIR, MODEL_DIR, VERIFY_DIR, WORK, write_json

import numpy as np
import torch
from transformers import AutoTokenizer

from kimodo.model.llm2vec import LLM2Vec
from streamed_model import StreamedBiModel
import verify_merge_functional as vmf

THIRD_PARTY = Path.home() / ".kimodo-venv" / "llm2vec-model"


def cos(r, x):
    return (r * x).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(x, axis=1))


def encode_runtime(path, dtype, texts):
    l2v = LLM2Vec.from_pretrained(base_model_name_or_path=str(path), peft_model_name_or_path=None, dtype=dtype,
                                  device_map="cuda:0", attn_implementation="sdpa")
    assert l2v.model.config._name_or_path == common.BASE_NAME
    out = np.stack([l2v.encode([t], batch_size=1, show_progress_bar=False, device="cuda:0").numpy()[0] for t in texts])
    del l2v
    gc.collect()
    torch.cuda.empty_cache()
    return out.astype(np.float32)


def main():
    ref = np.load(VERIFY_DIR / "ref_fp32.npz")
    texts = [str(t) for t in ref["texts"]]
    r = ref["reference"]
    res = {}
    embs = {}
    embs["nf4_fp16"] = encode_runtime(MODEL_DIR, torch.float16, texts)
    embs["nf4_bf16"] = encode_runtime(MODEL_DIR, torch.bfloat16, texts)
    if THIRD_PARTY.exists():
        embs["third_party_nf4_fp16"] = encode_runtime(THIRD_PARTY, torch.float16, texts)
        embs["third_party_nf4_bf16"] = encode_runtime(THIRD_PARTY, torch.bfloat16, texts)
    sm = StreamedBiModel(MERGED_DIR, dtype=torch.float16)
    sm.model.config._name_or_path = common.BASE_NAME
    l2v = LLM2Vec(model=sm.model, tokenizer=vmf.tokenizer(MERGED_DIR), **common.LLM2VEC_DEFAULTS)
    embs["fp16_compute_no_quant"] = vmf.encode_all(l2v, texts, "fp16_compute")
    del l2v, sm
    for k, x in embs.items():
        c = cos(r, x)
        res[k] = {"cos_mean": float(c.mean()), "cos_min": float(c.min()), "finite": bool(np.isfinite(x).all())}
    # 第三者版と自作版の一致度（同じ公式由来なら近いはず）
    if "third_party_nf4_fp16" in embs:
        c = cos(embs["third_party_nf4_fp16"], embs["nf4_fp16"])
        res["self_vs_third_party_fp16"] = {"cos_mean": float(c.mean()), "cos_min": float(c.min())}
    np.savez(VERIFY_DIR / "diagnose_nf4.npz", texts=np.array(texts), **embs)
    write_json(WORK / "logs" / "diagnose_nf4.json", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
