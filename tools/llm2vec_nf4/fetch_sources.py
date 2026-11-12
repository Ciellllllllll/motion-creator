"""公式配布元から固定リビジョンでファイルを取得し、配布メタデータと照合して source_manifest.json を作る。

実行: python -I fetch_sources.py <作業フォルダー>
- 認証は HF のローカル認証（hf auth login 済み）を使う。トークンを引数・ログに出さない。
- LFS ファイルは配布側 SHA-256、通常ファイルは Git blob ID（SHA-1）と照合する。
- Pickle 形式（original/*.pth）は取得しない。
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

SOURCES = {
    "base": {
        "repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        "dir": "meta-llama-3-8b-instruct",
        # 重み（Safetensors）・設定・トークナイザー・ライセンス。original/ の Pickle 重みは除外
        "include": lambda p: not p.startswith("original/") and not p.endswith((".pth", ".bin", ".pt")),
    },
    "mntp": {
        "repo": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        "revision": "31474e395ada192e8ed1586db6be79fb3b70c9c0",
        "dir": "llm2vec-mntp",
        "include": lambda p: not p.endswith((".pth", ".bin", ".pt")),
    },
    "supervised": {
        "repo": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        "revision": "baa8ebf04a1c2500e61288e7dad65e8ae42601a7",
        "dir": "llm2vec-supervised",
        "include": lambda p: not p.endswith((".pth", ".bin", ".pt")),
    },
}

# 動作生成モデルは既存キャッシュの使用中スナップショットを固定し、照合だけ行う
KIMODO = {
    "repo": "nvidia/Kimodo-SOMA-RP-v1",
    "revision": "defbe1f34f5fd031eb0ccfe19f3f92a9880c287b",
    "local": Path.home() / ".cache/huggingface/hub/models--nvidia--Kimodo-SOMA-RP-v1/snapshots"
             / "defbe1f34f5fd031eb0ccfe19f3f92a9880c287b",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check(local: Path, sibling) -> dict:
    size = local.stat().st_size
    digest = sha256(local)
    rec = {"bytes": size, "sha256": digest}
    if sibling.lfs is not None:
        rec["hub_kind"] = "lfs"
        rec["hub_sha256"] = sibling.lfs.sha256
        rec["match"] = digest == sibling.lfs.sha256 and size == sibling.lfs.size
    else:
        rec["hub_kind"] = "git"
        rec["hub_blob_id"] = sibling.blob_id
        rec["match"] = git_blob_sha1(local) == sibling.blob_id
    return rec


def main(work: Path):
    api = HfApi()
    inputs = work / "inputs"
    manifest = {"created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "note": "入力ファイルの固定版とハッシュ。match=配布メタデータと一致", "sources": {}, "files": []}
    ok = True
    for key, src in SOURCES.items():
        info = api.model_info(src["repo"], revision=src["revision"], files_metadata=True)
        assert info.sha == src["revision"], (info.sha, src["revision"])
        manifest["sources"][key] = {"repo": src["repo"], "revision": info.sha,
                                    "url": f"https://huggingface.co/{src['repo']}/tree/{info.sha}",
                                    "local_dir": str(inputs / src["dir"])}
        for s in sorted(info.siblings, key=lambda s: s.rfilename):
            if not src["include"](s.rfilename):
                manifest["files"].append({"source": key, "path": s.rfilename, "skipped": "pickle/original"})
                continue
            print(f"[{key}] {s.rfilename}", flush=True)
            local = Path(hf_hub_download(src["repo"], s.rfilename, revision=src["revision"],
                                         local_dir=inputs / src["dir"]))
            rec = check(local, s)
            rec.update(source=key, repo=src["repo"], revision=src["revision"], path=s.rfilename,
                       local_path=str(local),
                       retrieved_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"))
            ok &= rec["match"]
            print(f"    {rec['bytes']} bytes sha256={rec['sha256'][:16]}... match={rec['match']}", flush=True)
            manifest["files"].append(rec)

    info = api.model_info(KIMODO["repo"], revision=KIMODO["revision"], files_metadata=True)
    manifest["sources"]["kimodo"] = {"repo": KIMODO["repo"], "revision": info.sha,
                                     "local_dir": str(KIMODO["local"]),
                                     "note": "既存キャッシュ（使用中スナップショット）を照合のみ"}
    for s in sorted(info.siblings, key=lambda s: s.rfilename):
        local = KIMODO["local"] / s.rfilename
        if not local.exists():
            manifest["files"].append({"source": "kimodo", "path": s.rfilename, "missing_locally": True})
            continue
        rec = check(local, s)
        rec.update(source="kimodo", repo=KIMODO["repo"], revision=info.sha, path=s.rfilename,
                   local_path=str(local), retrieved_at="existing cache")
        ok &= rec["match"]
        print(f"[kimodo] {s.rfilename} match={rec['match']}", flush=True)
        manifest["files"].append(rec)

    manifest["all_match"] = bool(ok)
    (work / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("all_match:", ok)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
