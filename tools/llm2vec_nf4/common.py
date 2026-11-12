"""llm2vec_nf4 の共通設定。各スクリプトは import 前にオフライン化する。"""
import hashlib
import json
import os
from pathlib import Path

# 変換・検証中の暗黙ダウンロードを禁止する（仕様 6-6）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

WORK = Path(os.environ.get("LLM2VEC_NF4_WORK", r"D:\ModelBuild\llm2vec-nf4"))
INPUTS = WORK / "inputs"
BASE_DIR = INPUTS / "meta-llama-3-8b-instruct"
MNTP_DIR = INPUTS / "llm2vec-mntp"
SUP_DIR = INPUTS / "llm2vec-supervised"
MERGED_DIR = WORK / "merged-fp16"
OUT_NAME = "Llama-3-LLM2Vec-Kimodo-NF4-local-v1"
OUT_DIR = WORK / OUT_NAME
MODEL_DIR = OUT_DIR / "model"
VERIFY_DIR = WORK / "verify"
TOOLS = Path(__file__).resolve().parent

# Kimodo のプロンプト整形（LLM2Vec.prepare_for_tokenization）はこの名前で分岐する
BASE_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LLM2VEC_DEFAULTS = {"pooling_mode": "mean", "max_length": 512, "doc_max_length": 400, "skip_instruction": True}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def prompts():
    return load_json(TOOLS / "prompts.json")
