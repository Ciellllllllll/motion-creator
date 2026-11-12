"""自作 NF4 エンコーダーでポーズ集10種類を別フォルダーへ再生成する（既存ポーズ集は変更しない）。

実行: python -s regen_library.py [出力フォルダー] [id ...]
プロンプト・秒数・シードは tools/build_library.py の CLIPS と同じ。
index.json は既存と同じ形式に、自作エンコーダーの版とハッシュ（text_encoder）を加える。
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build_library import CLIPS, MODEL, bvh_duration  # noqa: E402

WORK = Path(os.environ.get("LLM2VEC_NF4_WORK", r"D:\ModelBuild\llm2vec-nf4"))
MODEL_DIR = WORK / "Llama-3-LLM2Vec-Kimodo-NF4-local-v1" / "model"
PY = WORK / "venv" / "Scripts" / "python.exe"
FORBIDDEN = ("llm2vec-model", "KIMODO-Meta3_llm2vec", "Kinisis", "falling back", "Text encoder service")


def generate(out: Path, cid, prompt, seconds, seed):
    log_path = out / "logs" / f"{cid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / cid
        cmd = [str(PY), "-s", str(HERE / "kimodo_gen_local.py"), prompt, "--model", MODEL,
               "--duration", str(seconds), "--seed", str(seed), "--output", str(stem), "--bvh"]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", KIMODO_NF4_MODEL_DIR=str(MODEL_DIR))
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        log_path.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"{cid}: kimodo_gen_local failed (see {log_path})")
        text = r.stdout + r.stderr
        marker = ("[CachedReferenceEncoder]" if os.environ.get("KIMODO_REFERENCE_EMB")
                  else "[LocalNF4Encoder] loading self-built model")
        if marker not in text:
            raise RuntimeError(f"{cid}: 想定したエンコーダーの読み込みログがない: {marker}")
        bad = [w for w in FORBIDDEN if w in text]
        if bad:
            raise RuntimeError(f"{cid}: 禁止された読み込み経路のログ: {bad}")
        (out / f"{cid}.bvh").write_bytes(stem.with_suffix(".bvh").read_bytes())
        (out / f"{cid}.npz").write_bytes(stem.with_suffix(".npz").read_bytes())


def main(out: Path, only):
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((MODEL_DIR.parent / "build_manifest.json").read_text(encoding="utf-8"))
    enc = {"name": manifest["name"], "model_dir": str(MODEL_DIR),
           "model_safetensors_sha256": manifest["outputs"]["model"]["model.safetensors"]["sha256"],
           "build_manifest_created_at": manifest["created_at"]}
    if os.environ.get("KIMODO_REFERENCE_EMB"):
        enc = {"name": "FP32 reference embeddings (comparison only, not for use)",
               "npz": os.environ["KIMODO_REFERENCE_EMB"]}
    for cid, name, prompt, seconds, seed, _ in CLIPS:
        if only and cid not in only:
            continue
        print("generating", cid, flush=True)
        generate(out, cid, prompt, seconds, seed)
    if not all((out / f"{c[0]}.bvh").exists() for c in CLIPS):
        print("一部のクリップが未生成のため index.json は書かない")
        return
    items = []
    for cid, name, prompt, seconds, seed, default_time in CLIPS:
        duration = bvh_duration(out / f"{cid}.bvh")
        items.append({"id": cid, "name": name, "file": f"{cid}.bvh", "duration": duration,
                      "default_time": min(default_time, duration), "static": False,
                      "prompt": prompt, "seed": seed, "model": MODEL, "text_encoder": enc})
    stand = next(i for i in items if i["id"] == "stand")
    base = dict(stand, id="base", name="基本姿勢", static=True)
    (out / "index.json").write_text(json.dumps({"items": [base] + items}, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print("wrote", out / "index.json")


if __name__ == "__main__":
    args = sys.argv[1:]
    out = Path(args.pop(0)) if args and not args[0].isidentifier() else WORK / "library-nf4-local-v1"
    main(out, set(args))
