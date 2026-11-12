"""量子化前の統合の正しさを文章特徴量で確認する（仕様 8.2 前半）。FP32（TF32無効）・1文ずつ・SDPA。

参照経路: 基礎モデル + 公式追加重み（PEFT の MNTP と supervised を未統合のまま同時に有効化）
比較対象: 保存した統合済み FP16 を FP32 で読み込んだもの
判別用  : MNTP のみ（supervised 欠落） / MNTP + supervised×2（二重適用）
参照経路の特徴量は verify/ref_fp32.npz に保存し、NF4 比較で再利用する。

FP32 全体は 64GB 機の仮想メモリに収まらないため streamed_model で層ごとに重みを読み込む。
LLM2Vec.encode は内部で self.to(device) を呼ぶため、encode と同じ関数
（_convert_to_str → prepare_for_tokenization → tokenize → forward）を1文ずつ呼ぶ。
"""
import gc
import json
import time

import common
from common import (BASE_DIR, BASE_NAME, LLM2VEC_DEFAULTS, MERGED_DIR, MNTP_DIR, SUP_DIR, VERIFY_DIR, WORK,
                    load_json, prompts, write_json)

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from kimodo.model.llm2vec import LLM2Vec
from kimodo.model.llm2vec.llm2vec import batch_to_device
from streamed_model import StreamedBiModel

HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
DEVICE = "cuda:0"


def tokenizer(d):
    tok = AutoTokenizer.from_pretrained(str(d))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def encode_one(l2v, text):
    """LLM2Vec.encode([text]) と同じ処理（instruction は空）。デバイス移動だけしない。"""
    s = l2v._convert_to_str("", text)
    features = l2v.tokenize([l2v.prepare_for_tokenization(s)])
    assert int(features["attention_mask"].sum()) == features["attention_mask"].numel(), "パディングが入った"
    features = batch_to_device(features, DEVICE)
    with torch.no_grad():
        emb = l2v.forward(features).detach().cpu().to(torch.float32)
    return emb[0].numpy()


def encode_all(l2v, texts, label):
    assert l2v.model.config._attn_implementation == "sdpa"
    assert l2v.prepare_for_tokenization("x").startswith(HEADER), "Llama 3 のプロンプト整形が効いていない"
    l2v.eval()
    t0 = time.time()
    out = np.stack([encode_one(l2v, t) for t in texts]).astype(np.float32)
    print(f"{label}: {len(texts)} prompts {time.time() - t0:.0f}s", flush=True)
    return out


def metrics(ref, x):
    cos = (ref * x).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(x, axis=1))
    rel = np.linalg.norm(ref - x, axis=1) / np.linalg.norm(ref, axis=1)
    return {"cos_min": float(cos.min()), "cos_mean": float(cos.mean()), "rel_l2_max": float(rel.max()),
            "finite": bool(np.isfinite(x).all()), "shape": list(x.shape), "cos": [float(c) for c in cos]}


def main():
    tol = load_json(common.TOOLS / "tolerances.json")["merge_functional"]
    p = prompts()
    texts = p["library"] + p["extra"]
    report = {"prompts": len(texts), "tolerance": tol,
              "method": "streamed_model(FP32, GPU, TF32無効, 層ごとに重みを読み込み), 1文ずつ, SDPA, パディングなし",
              "reference_definition": "基礎モデル + PEFT LoRA(mntp, supervised) を未統合で同時に有効化"}

    # トークナイザーの一致確認（基礎モデル vs MNTP リポジトリ同梱）
    tb, tm = tokenizer(BASE_DIR), tokenizer(MNTP_DIR)
    fmt = [HEADER + t.strip() + "<|eot_id|>" for t in texts]
    same = [tb(x)["input_ids"] == tm(x)["input_ids"] for x in fmt]
    report["tokenizer_base_vs_mntp_identical_ids"] = all(same)
    report["tokenizer_bos_first"] = all(tb(x)["input_ids"][0] == tb.bos_token_id for x in fmt)
    assert all(same), "基礎モデルと MNTP のトークナイザーでトークン列が異なる"

    embs = {}
    # 参照経路と判別用（同じ基礎モデルで、有効にする追加重みだけを切り替える）
    sm = StreamedBiModel(BASE_DIR)
    sm.model.config._name_or_path = BASE_NAME
    peft = PeftModel.from_pretrained(sm.model, str(MNTP_DIR), adapter_name="mntp")
    peft.load_adapter(str(SUP_DIR), adapter_name="supervised")
    peft.load_adapter(str(SUP_DIR), adapter_name="supervised_again")
    l2v = LLM2Vec(model=peft, tokenizer=tb, **LLM2VEC_DEFAULTS)
    for label, active in (("reference", ["mntp", "supervised"]), ("mntp_only", ["mntp"]),
                          ("double_supervised", ["mntp", "supervised", "supervised_again"])):
        peft.base_model.set_adapter(active)
        assert sorted(peft.base_model.active_adapters) == sorted(active)
        embs[label] = encode_all(l2v, texts, label)
    del l2v, peft, sm
    gc.collect()
    torch.cuda.empty_cache()

    # 統合済み（config.json の _name_or_path と llm2vec_config.json を LLM2Vec.from_pretrained と同様に反映）
    sm = StreamedBiModel(MERGED_DIR)
    cfg = load_json(MERGED_DIR / "config.json")
    sm.model.config._name_or_path = cfg["_name_or_path"]
    assert sm.model.config._name_or_path == BASE_NAME
    assert not sm.unexpected, sm.unexpected
    l2v_cfg = load_json(MERGED_DIR / "llm2vec_config.json")
    assert l2v_cfg == LLM2VEC_DEFAULTS, l2v_cfg
    report["merged_llm2vec_settings"] = l2v_cfg
    l2v = LLM2Vec(model=sm.model, tokenizer=tokenizer(MERGED_DIR), **l2v_cfg)
    embs["merged"] = encode_all(l2v, texts, "merged")
    del l2v, sm
    gc.collect()

    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(VERIFY_DIR / "ref_fp32.npz", texts=np.array(texts), **embs)

    def ok(m):
        return m["finite"] and m["cos_min"] >= tol["cosine_min_ge"] and m["rel_l2_max"] <= tol["relative_l2_max_le"]

    for k in ("merged", "mntp_only", "double_supervised"):
        m = metrics(embs["reference"], embs[k])
        m["within_tolerance"] = ok(m)
        report[k] = m
    report["reference_finite"] = bool(np.isfinite(embs["reference"]).all())
    report["pass"] = (report["merged"]["within_tolerance"] and not report["mntp_only"]["within_tolerance"]
                      and not report["double_supervised"]["within_tolerance"] and report["reference_finite"])
    write_json(WORK / "logs" / "verify_merge_functional.json", report)
    summary = {k: ({kk: v for kk, v in report[k].items() if kk != "cos"} if isinstance(report[k], dict) else report[k])
               for k in report}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
