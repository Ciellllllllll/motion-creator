"""検証結果を成果物フォルダーの verify_report.json にまとめる。認証情報の混入も検査する。

実行: python -s make_report.py
"""
import datetime as dt
import json
import re
from pathlib import Path

from common import OUT_DIR, TOOLS, WORK, load_json, write_json

LOGS = WORK / "logs"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}")


def scan_credentials():
    roots = [LOGS, OUT_DIR, TOOLS, WORK / "library-nf4-local-v1", WORK / "addon-check" / "out", WORK / "backup"]
    hits, scanned = [], 0
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".json", ".log", ".txt", ".md", ".py", ".diff", ""):
                scanned += 1
                if TOKEN.search(p.read_bytes()):
                    hits.append(str(p))
    return {"pattern": "hf_ + 20文字以上", "files_scanned": scanned, "hits": hits, "pass": not hits}


def main():
    blender = load_json(WORK / "addon-check" / "out" / "blender_check.json")
    report = {
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tolerances": load_json(TOOLS / "tolerances.json"),
        "provenance": {"source_manifest_all_match": load_json(WORK / "source_manifest.json")["all_match"],
                       "third_party_weights_used_in_build": False,
                       "note": "入力は公式3リポジトリの固定リビジョンのみ（source_manifest.json）。第三者 NF4 は diagnose_nf4 の比較でだけ読み込んだ"},
        "bidirectional_attention": load_json(LOGS / "bidirectional_check.json"),
        "merge_weights": load_json(LOGS / "verify_merge_weights.json"),
        "merge_functional": {k: v for k, v in load_json(LOGS / "verify_merge_functional.json").items()},
        "nf4": load_json(LOGS / "verify_nf4.json"),
        "nf4_diagnosis": load_json(LOGS / "diagnose_nf4.json"),
        "nf4_discrimination": load_json(LOGS / "nf4_discrimination.json"),
        "kimodo_generation": {
            "kimodo_model": "nvidia/Kimodo-SOMA-RP-v1 @ defbe1f34f5fd031eb0ccfe19f3f92a9880c287b（重み SHA-256 照合）",
            "encoder_log_marker": "[LocalNF4Encoder] loading self-built model（全10本のログで確認、禁止語なし）",
            "library_dir": str(WORK / "library-nf4-local-v1"),
            "machine_check_vs_original": load_json(LOGS / "check_library_nf4_vs_original.json"),
            "machine_check_vs_fp32_reference": load_json(LOGS / "check_library_nf4_vs_reference.json"),
        },
        "blender": {
            "addon_pipeline": blender,
            "visual_check": {
                "clip_frames": str(WORK / "addon-check" / "out" / "contact_sheet.png"),
                "fbx_playback": str(WORK / "addon-check" / "out" / "fbx_strips.png"),
                "result": "2026-11-12 目視確認: 10種類とも動作の意味・左右（手を振るは右手）・姿勢・接地に破綻なし。"
                          "小走りは片手を胸元に寄せた姿勢が続く（旧ポーズ集でも同じ動き、腰軌跡差 0.027 m）",
            },
        },
        "switch_and_restore": {
            "switched": "motionbricks_blender/library を自作 NF4 で再生成したものへ置換（.bvh 10 + index.json）",
            "tests_after_switch": "pytest tests: 14 passed / tests/gpu/test_library.py: 5 passed",
            "restore_check": "backup/library-original を戻し、元の SHA-256 と 11/11 一致、git 差分 0、test_library 5 passed",
            "re_switch_check": "自作版へ戻し 11/11 一致、test_library 5 passed",
        },
        "credentials_in_logs": scan_credentials(),
    }
    report["pass"] = bool(report["provenance"]["source_manifest_all_match"] and report["merge_weights"]["pass"]
                          and report["merge_functional"]["pass"] and report["nf4"]["pass"]
                          and report["kimodo_generation"]["machine_check_vs_original"]["all_pass"]
                          and report["credentials_in_logs"]["pass"])
    write_json(OUT_DIR / "verify_report.json", report)
    print("pass:", report["pass"], "credentials:", report["credentials_in_logs"])


if __name__ == "__main__":
    main()
