"""自作 NF4 版 LLM2Vec を明示パスから読み込む Kimodo 用テキストエンコーダー。

- 読み込み先は引数 model_dir（または環境変数 KIMODO_NF4_MODEL_DIR）だけ。見つからなければ停止し、
  Aero-Ex 版・Kinisis 版・Hugging Face への代替読み込み（フォールバック）はしない。
- build_manifest.json に記録した出力ハッシュと model.safetensors の SHA-256 が一致しなければ停止する。
- 公式 nv-tlabs/kimodo の LLM2VecEncoder と同じく1文ずつ（batch_size=1）エンコードする。
  この1文ずつ・SDPA の条件で LlamaBiModel は双方向になる（check_bidirectional.py で実測）。
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from kimodo.model.llm2vec import LLM2Vec

BASE_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_dir(model_dir: Path) -> dict:
    """model_dir の重みが build_manifest.json の記録と一致するか確かめ、記録を返す。"""
    manifest_path = model_dir.parent / "build_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"build_manifest.json がない: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["outputs"]["model"]
    for name, rec in expected.items():
        if not name.endswith(".safetensors"):
            continue
        actual = _sha256(model_dir / name)
        if actual != rec["sha256"]:
            raise RuntimeError(f"{name} のハッシュが記録と違う: {actual} != {rec['sha256']}")
    return manifest


class CachedReferenceEncoder:
    """比較専用: verify/ref_fp32.npz の全精度参照特徴量を返す（NF4 化の影響を動作で測るため）。
    登録されていない文章が来たら停止する。採用・切り替えの対象ではない。"""

    def __init__(self, npz_path: str, llm_dim: int = 4096, **_ignored) -> None:
        d = np.load(npz_path)
        self.table = {str(t).strip(): d["reference"][i] for i, t in enumerate(d["texts"])}
        self.llm_dim = llm_dim
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[CachedReferenceEncoder] FP32 reference embeddings from {npz_path} ({len(self.table)} texts)",
              flush=True)

    def to(self, device):
        return self

    def eval(self):
        return self

    def get_device(self):
        return torch.device(self._device)

    def __call__(self, text):
        is_string = isinstance(text, str)
        if is_string:
            text = [text]
        missing = [t for t in text if t.strip() not in self.table]
        if missing:
            raise KeyError(f"参照特徴量がない文章: {missing}")
        encoded = np.stack([self.table[t.strip()] for t in text]).astype(np.float32)[:, None]
        lengths = np.ones(len(encoded), dtype=int).tolist()
        if is_string:
            encoded, lengths = encoded[0], lengths[0]
        return torch.as_tensor(encoded).to(self._device), lengths


class LocalNF4Encoder:
    """Kimodo の text_encoder として使う。呼び出し形式は公式 LLM2VecEncoder と同じ。"""

    def __init__(self, model_dir: str | None = None, dtype: str = "float16", llm_dim: int = 4096,
                 device: str = "auto", verify_hash: bool = True, **_ignored) -> None:
        model_dir = model_dir or os.environ.get("KIMODO_NF4_MODEL_DIR")
        if not model_dir:
            raise RuntimeError("自作 NF4 モデルのパスが未指定（KIMODO_NF4_MODEL_DIR）。代替モデルは使わない")
        self.model_dir = Path(model_dir).resolve()
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(f"自作 NF4 モデルが見つからない: {self.model_dir}")
        cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        qc = cfg.get("quantization_config", {})
        if not (qc.get("load_in_4bit") and qc.get("bnb_4bit_quant_type") == "nf4"):
            raise RuntimeError(f"NF4 量子化モデルではない: {self.model_dir}")
        if cfg.get("_name_or_path") != BASE_NAME:
            raise RuntimeError(f"_name_or_path が {BASE_NAME} でない（プロンプト整形が変わる）")
        self.manifest = verify_model_dir(self.model_dir) if verify_hash else None
        self.llm_dim = llm_dim
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = device
        tag = self.manifest["outputs"]["model"]["model.safetensors"]["sha256"][:16] if self.manifest else "unverified"
        print(f"[LocalNF4Encoder] loading self-built model: {self.model_dir} (model.safetensors sha256 {tag}...)",
              flush=True)
        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=str(self.model_dir),
            peft_model_name_or_path=None,
            dtype=getattr(torch, dtype),
            device_map=device,
            attn_implementation="sdpa",
        )
        assert self.model.model.config._name_or_path == BASE_NAME
        assert self.model.model.config._attn_implementation == "sdpa"
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"[LocalNF4Encoder] loaded on {self.get_device()}", flush=True)

    def to(self, device):
        # 4bit 重みは読み込み時のデバイスに固定する（bitsandbytes は .to() での移動に制約がある）
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return self.model.model.device

    def __call__(self, text: list[str] | str):
        is_string = isinstance(text, str)
        if is_string:
            text = [text]
        with torch.no_grad():
            encoded = self.model.encode(text, batch_size=1, show_progress_bar=False,
                                        device=str(self.get_device()))
        assert self.llm_dim == encoded.shape[-1]
        encoded = encoded[:, None]
        lengths = np.ones(len(encoded), dtype=int).tolist()
        if is_string:
            encoded, lengths = encoded[0], lengths[0]
        return torch.as_tensor(encoded).to(self.get_device()), lengths
