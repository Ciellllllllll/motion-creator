"""自作 NF4 エンコーダーで kimodo_gen を実行する起動スクリプト（Kimodo のソースは変更しない）。

実行: python -s kimodo_gen_local.py "<prompt>" --model Kimodo-SOMA-RP-v1 --duration 4 --seed 11 --output out --bvh
必須環境変数: KIMODO_NF4_MODEL_DIR（自作 NF4 の model/ フォルダー）
任意: KIMODO_CHECKPOINT_DIR（既定: <作業フォルダー>/kimodo-checkpoints。中に Kimodo-SOMA-RP-v1 を置く）

- Kimodo の TEXT_ENCODER_PRESETS に自作エンコーダーを1件追加し、TEXT_ENCODER_MODE=local で選ぶ。
  API サーバーや既存 LLM2Vec（Aero-Ex 版の直書きパス）へのフォールバック経路を通らない。
- 動作生成モデルは CHECKPOINT_DIR から固定スナップショットを読み、HF へは接続しない。
"""
import hashlib
import importlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIMODO_REVISION = "defbe1f34f5fd031eb0ccfe19f3f92a9880c287b"  # 重みの SHA-256 は source_manifest.json から読む


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TEXT_ENCODER_MODE"] = "local"
    os.environ["TEXT_ENCODER"] = "llm2vec_local_nf4"
    os.environ.pop("TEXT_ENCODER_URL", None)
    model_dir = os.environ.get("KIMODO_NF4_MODEL_DIR")
    if not model_dir or not (Path(model_dir) / "config.json").exists():
        sys.exit(f"KIMODO_NF4_MODEL_DIR が未設定か、モデルがない: {model_dir!r}")
    work = Path(os.environ.get("LLM2VEC_NF4_WORK", r"D:\ModelBuild\llm2vec-nf4"))
    ckpt = Path(os.environ.get("KIMODO_CHECKPOINT_DIR", work / "kimodo-checkpoints"))
    snap = ckpt / "Kimodo-SOMA-RP-v1"
    if not (snap / "config.yaml").exists():
        sys.exit(f"Kimodo の固定スナップショットがない: {snap}")
    os.environ["CHECKPOINT_DIR"] = str(ckpt)

    # 動作生成モデルの重みが固定版と一致するか確認
    import json
    src = json.loads((Path(model_dir).parent / "source_manifest.json").read_text(encoding="utf-8"))
    rec = next(f for f in src["files"] if f.get("source") == "kimodo" and f.get("path") == "model.safetensors")
    h = hashlib.sha256()
    with open(snap / "model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != rec["sha256"] or rec["revision"] != KIMODO_REVISION:
        sys.exit("Kimodo-SOMA-RP-v1 の重みが固定版と一致しない")
    print(f"[kimodo_gen_local] Kimodo-SOMA-RP-v1 @ {KIMODO_REVISION} sha256 {rec['sha256'][:16]}... ({snap})",
          flush=True)

    sys.path.insert(0, str(HERE))
    lm = importlib.import_module("kimodo.model.load_model")
    ref = os.environ.get("KIMODO_REFERENCE_EMB")
    if ref:  # 比較専用: 全精度の参照特徴量で生成し、NF4 化が動作に与える影響を測る
        lm.TEXT_ENCODER_PRESETS["llm2vec_local_nf4"] = {
            "target": "local_nf4_encoder.CachedReferenceEncoder", "kwargs": {"npz_path": ref, "llm_dim": 4096}}
    else:
        lm.TEXT_ENCODER_PRESETS["llm2vec_local_nf4"] = {
            "target": "local_nf4_encoder.LocalNF4Encoder",
            "kwargs": {"model_dir": str(Path(model_dir).resolve()), "dtype": "float16", "llm_dim": 4096},
        }
    from kimodo.scripts.generate import main as gen_main
    gen_main()


if __name__ == "__main__":
    main()
