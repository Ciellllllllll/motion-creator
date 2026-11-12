"""成果物フォルダーに build_manifest.json・source_manifest.json・licenses/・NOTICE を置く。

実行: python -s make_manifest.py   （quantize.py の後、verify_nf4.py の前）
"""
import datetime as dt
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import common
from common import BASE_DIR, MNTP_DIR, MODEL_DIR, OUT_DIR, OUT_NAME, SUP_DIR, TOOLS, WORK, load_json, write_json

import torch

SRC = WORK / "src"
KIMODO_SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--nvidia--Kimodo-SOMA-RP-v1/snapshots"
                   / "defbe1f34f5fd031eb0ccfe19f3f92a9880c287b")

NOTICE = """\
{name}

Built with Meta Llama 3

Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

このモデルは、次の公式配布物だけから、ローカルで統合・4bit NF4 量子化して作成した。
第三者が変換・配布した重みは含まない。各入力の固定版とハッシュは source_manifest.json、
作成手順・環境・出力ハッシュは build_manifest.json を参照。

構成要素と出所:
- 基礎モデル: meta-llama/Meta-Llama-3-8B-Instruct @ 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2
  Meta Llama 3 Community License（licenses/Meta-Llama-3-LICENSE）と
  Acceptable Use Policy（licenses/Meta-Llama-3-USE_POLICY.md）に従う。
- MNTP 追加重み: McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp @ 31474e395ada192e8ed1586db6be79fb3b70c9c0
  （配布ページの表記: MIT。Llama 3 由来部分には上記 Meta の条件が引き続き適用される）
- supervised 追加重み: McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised @ baa8ebf04a1c2500e61288e7dad65e8ae42601a7
  （配布ページの表記: MIT。同上）
- LLM2Vec / 双方向 Llama 実装: McGill-NLP/llm2vec（MIT, licenses/llm2vec-LICENSE）を
  nv-tlabs/kimodo（Apache-2.0, licenses/kimodo-LICENSE）が取り込んだもの。
- 実行時: nv-tlabs/kimodo の派生 Aero-Ex/kimodo @ 3b45cc6d420136bfa6128e8be5e81e45521f2b7a（Apache-2.0）、
  動作生成モデル nvidia/Kimodo-SOMA-RP-v1 @ defbe1f34f5fd031eb0ccfe19f3f92a9880c287b
  （licenses/Kimodo-SOMA-RP-v1-LICENSE）。
- 変換ツール: transformers 5.1.0 / peft 0.21.2（Apache-2.0）、bitsandbytes 0.50.2（MIT）、safetensors 0.8.0（Apache-2.0）。

自作・軽量化によってライセンスは MIT 等に変わらない。外部へ提供・再配布する場合は、
その提供形態に合わせて Meta Llama 3 Community License の条件（契約書の同梱、
"Built with Meta Llama 3" の表示、本 NOTICE の同梱など）を確認すること。
"""


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lic = OUT_DIR / "licenses"
    lic.mkdir(exist_ok=True)
    copies = {
        BASE_DIR / "LICENSE": "Meta-Llama-3-LICENSE",
        BASE_DIR / "USE_POLICY.md": "Meta-Llama-3-USE_POLICY.md",
        SRC / "mcgill-llm2vec" / "LICENSE": "llm2vec-LICENSE",
        SRC / "nv-kimodo" / "LICENSE": "kimodo-LICENSE",
        SRC / "aero-kimodo" / "LICENSE": "Aero-Ex-kimodo-LICENSE",
        KIMODO_SNAPSHOT / "LICENSE": "Kimodo-SOMA-RP-v1-LICENSE",
        MNTP_DIR / "README.md": "LLM2Vec-mntp-README.md",
        SUP_DIR / "README.md": "LLM2Vec-supervised-README.md",
    }
    for src, name in copies.items():
        shutil.copyfile(src, lic / name)
    (OUT_DIR / "NOTICE").write_text(NOTICE.format(name=OUT_NAME), encoding="utf-8")
    shutil.copyfile(WORK / "source_manifest.json", OUT_DIR / "source_manifest.json")

    import bitsandbytes, peft, safetensors, transformers
    import kimodo
    kimodo_l2v = Path(kimodo.__file__).parent / "model" / "llm2vec"
    official = SRC / "nv-kimodo" / "kimodo" / "model" / "llm2vec"
    impl = {}
    for rel in ("llm2vec.py", "models/bidirectional_llama.py", "models/utils.py", "models/attn_mask_utils.py"):
        a, b = common.sha256(kimodo_l2v / rel), common.sha256(official / rel)
        impl[rel] = {"sha256": a, "identical_to_nv_tlabs_official": a == b}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    repo = TOOLS.parents[1]
    manifest = {
        "name": OUT_NAME,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "code": {
            "motion_creator_commit": git(repo, "rev-parse", "HEAD"),
            "motion_creator_dirty_paths": git(repo, "status", "--porcelain", "--", "tools/llm2vec_nf4").splitlines(),
            "scripts_sha256": {p.name: common.sha256(p) for p in sorted(TOOLS.iterdir()) if p.is_file()},
            "reference_repos": {d.name: {"url": git(d, "remote", "get-url", "origin"), "commit": git(d, "rev-parse", "HEAD")}
                                for d in sorted(SRC.iterdir()) if (d / ".git").exists()},
            "aero_kimodo_reference_commit_for_procedure": "bfc2dc4 (bake_model.py / quantize_baked.py の処理順序のみ参照)",
            "llm2vec_implementation": {"package": "kimodo.model.llm2vec (installed Aero-Ex/kimodo 3b45cc6)",
                                       "files": impl},
        },
        "environment": {
            "os": platform.platform(), "python": sys.version.split()[0],
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": smi, "transformers": transformers.__version__, "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__, "safetensors": safetensors.__version__,
            "lock_file": "tools/llm2vec_nf4/requirements.lock.txt",
            "lock_file_sha256": common.sha256(TOOLS / "requirements.lock.txt"),
        },
        "inputs": {"source_manifest_sha256": common.sha256(WORK / "source_manifest.json"),
                   "sources": load_json(WORK / "source_manifest.json")["sources"]},
        "merge": load_json(WORK / "logs" / "merge_log.json"),
        "quantization": load_json(WORK / "logs" / "quantize_log.json"),
        "outputs": {"model": {p.name: {"bytes": p.stat().st_size, "sha256": common.sha256(p)}
                              for p in sorted(MODEL_DIR.iterdir()) if p.is_file()}},
        "offline": "変換・検証は HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1、trust_remote_code 不使用",
    }
    assert all(v["identical_to_nv_tlabs_official"] for v in impl.values()), impl
    write_json(OUT_DIR / "build_manifest.json", manifest)
    print("wrote", OUT_DIR / "build_manifest.json")


if __name__ == "__main__":
    main()
