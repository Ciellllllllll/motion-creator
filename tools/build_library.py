"""Kimodo でポーズ集を生成する（開発時のみ。利用時に Kimodo は不要）。

実行：python tools/build_library.py [id ...]
Kimodo の BVH は標準 T ポーズ「オフ」（SOMA ネイティブ形式）で書き出す。
MotionBricks の骨格と関節・基本姿勢が一致するので、そのまま読み込める。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL = "Kimodo-SOMA-RP-v1"
LIB = Path(__file__).resolve().parents[1] / "motionbricks_blender" / "library"
# 既定: 公式ソースから自作した NF4 エンコーダー（tools/llm2vec_nf4、ハッシュ照合つき）で生成する。
# KIMODO_GEN を指定したときだけ、その kimodo_gen 実行ファイルを使う。
NF4_WORK = Path(os.environ.get("LLM2VEC_NF4_WORK", r"D:\ModelBuild\llm2vec-nf4"))
NF4_MODEL_DIR = NF4_WORK / "Llama-3-LLM2Vec-Kimodo-NF4-local-v1" / "model"
KIMODO_GEN = os.environ.get("KIMODO_GEN")


def kimodo_command():
    if KIMODO_GEN:
        return [KIMODO_GEN], {}
    launcher = Path(__file__).resolve().parent / "llm2vec_nf4" / "kimodo_gen_local.py"
    return ([str(NF4_WORK / "venv" / "Scripts" / "python.exe"), "-s", str(launcher)],
            {"KIMODO_NF4_MODEL_DIR": str(NF4_MODEL_DIR)})

# (id, 日本語名, プロンプト, 秒, シード, 既定の時刻)
CLIPS = [
    ("stand", "立ち", "A person stands still in a relaxed neutral pose with arms at their sides.", 4.0, 11, 2.0),
    ("walk", "歩き", "A person walks forward at a normal pace.", 5.0, 12, 2.5),
    ("jog", "小走り", "A person jogs forward at a light pace.", 4.0, 13, 2.0),
    ("run", "走り", "A person runs forward fast.", 4.0, 14, 2.0),
    ("crouch", "しゃがみ", "A person crouches down low and stays crouched.", 4.0, 15, 3.0),
    ("crouch_walk", "しゃがみ歩き", "A person walks forward while crouching low.", 5.0, 16, 2.5),
    ("jump", "ジャンプ", "A person jumps straight up in place and lands.", 4.0, 17, 1.2),
    ("sit", "座る", "A person sits down on a chair and stays seated.", 5.0, 18, 4.0),
    ("wave", "手を振る", "A person stands and waves with the right hand.", 4.0, 19, 2.0),
    ("turn", "振り返る", "A person standing turns around to face the opposite direction.", 4.0, 20, 2.0),
]


def bvh_duration(path: Path) -> float:
    frames = dt = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("Frames:"):
                frames = int(line.split()[1])
            elif line.startswith("Frame Time:"):
                dt = float(line.split()[2])
                break
    return round((frames - 1) * dt, 4)


def generate(cid, prompt, seconds, seed):
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / cid
        base, extra_env = kimodo_command()
        cmd = base + [prompt, "--model", MODEL, "--duration", str(seconds), "--seed", str(seed),
                      "--output", str(stem), "--bvh"]
        subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONIOENCODING="utf-8", **extra_env))
        shutil.copyfile(stem.with_suffix(".bvh"), LIB / f"{cid}.bvh")


def main(only):
    LIB.mkdir(parents=True, exist_ok=True)
    for cid, name, prompt, seconds, seed, _ in CLIPS:
        if only and cid not in only:
            continue
        if not only and (LIB / f"{cid}.bvh").exists():
            continue
        print("generating", cid, flush=True)
        generate(cid, prompt, seconds, seed)
    enc = None
    if not KIMODO_GEN:
        manifest = json.loads((NF4_MODEL_DIR.parent / "build_manifest.json").read_text(encoding="utf-8"))
        enc = {"name": manifest["name"], "model_dir": str(NF4_MODEL_DIR),
               "model_safetensors_sha256": manifest["outputs"]["model"]["model.safetensors"]["sha256"],
               "build_manifest_created_at": manifest["created_at"]}
    items = []
    for cid, name, prompt, seconds, seed, default_time in CLIPS:
        duration = bvh_duration(LIB / f"{cid}.bvh")
        item = {"id": cid, "name": name, "file": f"{cid}.bvh", "duration": duration,
                "default_time": min(default_time, duration), "static": False,
                "prompt": prompt, "seed": seed, "model": MODEL}
        if enc:
            item["text_encoder"] = enc
        items.append(item)
    stand = next(i for i in items if i["id"] == "stand")
    base = dict(stand, id="base", name="基本姿勢", static=True)
    (LIB / "index.json").write_text(json.dumps({"items": [base] + items}, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print("wrote", LIB / "index.json")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
