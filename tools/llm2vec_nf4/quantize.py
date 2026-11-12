"""統合済み FP16 モデルを GPU で NF4（二重量子化・計算型 FP16）に変換し、Safetensors で保存する。

実行: python -s quantize.py
保存後はこのプロセスを終了し、検証は verify_nf4.py を別プロセスで実行する（仕様 7.2）。
"""
import datetime as dt
import shutil
import time

import common
from common import BASE_NAME, MERGED_DIR, MODEL_DIR, WORK, load_json, write_json

import torch
from transformers import BitsAndBytesConfig

from kimodo.model.llm2vec import LLM2Vec

QUANT = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
             bnb_4bit_compute_dtype=torch.float16)


def main():
    log = {"started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "input": str(MERGED_DIR), "quantization": {k: str(v) for k, v in QUANT.items()}}
    t0 = time.time()
    l2v = LLM2Vec.from_pretrained(str(MERGED_DIR), quantization_config=BitsAndBytesConfig(**QUANT),
                                  dtype=torch.float16, device_map="cuda:0", attn_implementation="sdpa")
    log["load_seconds"] = round(time.time() - t0, 1)
    log["cuda_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    n4 = sum(1 for m in l2v.model.modules() if type(m).__name__ == "Linear4bit")
    log["linear4bit_modules"] = n4
    assert n4 == 32 * 7, n4
    assert l2v.model.config._name_or_path == BASE_NAME

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True)
    l2v.model.save_pretrained(str(MODEL_DIR), safe_serialization=True, max_shard_size="20GB")
    # トークナイザーは統合済みフォルダー（= 公式ファイルのコピー）からそのまま持ち込む
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "llm2vec_config.json"):
        shutil.copyfile(MERGED_DIR / name, MODEL_DIR / name)

    cfg = load_json(MODEL_DIR / "config.json")
    if cfg.get("_name_or_path") != BASE_NAME:
        cfg["_name_or_path"] = BASE_NAME
        write_json(MODEL_DIR / "config.json", cfg)
        log["config_name_or_path_patched"] = True
    qc = cfg["quantization_config"]
    assert qc["load_in_4bit"] and qc["bnb_4bit_quant_type"] == "nf4" and qc["bnb_4bit_use_double_quant"]
    assert qc["bnb_4bit_compute_dtype"] in ("float16", "torch.float16"), qc["bnb_4bit_compute_dtype"]
    assert not list(MODEL_DIR.glob("*.bin")) and not list(MODEL_DIR.glob("*.pt")), "Pickle 形式が出力された"
    log["saved_quantization_config"] = qc
    log["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": common.sha256(p)}
                    for p in sorted(MODEL_DIR.iterdir()) if p.is_file()}
    log["finished_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(WORK / "logs" / "quantize_log.json", log)
    print("saved NF4 ->", MODEL_DIR)


if __name__ == "__main__":
    main()
