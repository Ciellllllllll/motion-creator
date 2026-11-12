"""統合済み FP16 重みを、配布ファイルから独立に計算した期待値と1テンソルずつ照合する。

期待値（FP32）: W = W0 + s1*B1@A1 + s2*B2@A2   （s = lora_alpha / r = 2）
- LoRA 対象外のテンソルは W0 の FP16 化と一致すること。
- 許容: 各要素 |merged - W| <= max(|W| * 2^-11, 2^-24)（FP16 への1回の丸め）
- 判別力: supervised の差分 ||s2*B2@A2|| が丸め誤差上限ノルムの10倍以上（欠落・二重適用なら必ず不合格）
"""
import json

import common
from common import BASE_DIR, LORA_TARGETS, MERGED_DIR, MNTP_DIR, SUP_DIR, WORK, load_json, write_json

import torch
from safetensors import safe_open


def open_sharded(d):
    """(weight_map, getter)。Windows で safe_open はファイルサイズ分のコミットを使うので、
    テンソルごとに開いて閉じる（全シャードを同時に開かない）。"""
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        wm = load_json(idx)["weight_map"]
    else:
        with safe_open(d / "model.safetensors", "pt") as f:
            wm = {k: "model.safetensors" for k in f.keys()}

    def get(key):
        with safe_open(d / wm[key], "pt") as f:
            return f.get_tensor(key).clone()
    return wm, get


def main():
    tol = load_json(common.TOOLS / "tolerances.json")["weight_merge"]
    s = {n: load_json(d / "adapter_config.json") for n, d in (("mntp", MNTP_DIR), ("sup", SUP_DIR))}
    scale = {n: c["lora_alpha"] / c["r"] for n, c in s.items()}
    ad = {"mntp": safe_open(MNTP_DIR / "adapter_model.safetensors", "pt"),
          "sup": safe_open(SUP_DIR / "adapter_model.safetensors", "pt")}
    base_wm, base_get = open_sharded(BASE_DIR)
    m_wm, m_get = open_sharded(MERGED_DIR)

    base_keys = {k[len("model."):]: k for k in base_wm if k.startswith("model.")}
    assert set(m_wm) == set(base_keys), (set(m_wm) ^ set(base_keys))
    report = {"tensors": 0, "lora_tensors": 0, "violations": 0, "min_discrimination_ratio": None,
              "max_abs_err": 0.0, "worst": None, "dtype_merged": set()}
    min_ratio = float("inf")
    for name in sorted(m_wm):
        merged = m_get(name)
        report["dtype_merged"].add(str(merged.dtype))
        merged = merged.to(torch.float32)
        w = base_get(base_keys[name]).to(torch.float32)
        mod = name[: -len(".weight")] if name.endswith(".weight") else None
        is_lora = mod is not None and mod.split(".")[-1] in LORA_TARGETS
        if is_lora:
            deltas = {}
            for n in ("mntp", "sup"):
                a = ad[n].get_tensor(f"base_model.model.{mod}.lora_A.weight").to(torch.float32)
                b = ad[n].get_tensor(f"base_model.model.{mod}.lora_B.weight").to(torch.float32)
                deltas[n] = scale[n] * (b @ a)
            w = w + deltas["mntp"] + deltas["sup"]
            report["lora_tensors"] += 1
        bound = torch.clamp(w.abs() * 2.0 ** -11, min=2.0 ** -24)
        err = (merged - w).abs()
        viol = int((err > bound).sum())
        report["violations"] += viol
        mx = float(err.max())
        if mx > report["max_abs_err"]:
            report["max_abs_err"], report["worst"] = mx, name
        if is_lora:
            ratio = float(deltas["sup"].norm() / bound.norm())
            min_ratio = min(min_ratio, ratio)
        report["tensors"] += 1
    report["min_discrimination_ratio"] = min_ratio
    report["dtype_merged"] = sorted(report["dtype_merged"])
    report["expected_lora_tensors"] = 32 * len(LORA_TARGETS)
    report["tolerance"] = tol
    report["pass"] = (report["violations"] == 0 and report["lora_tensors"] == report["expected_lora_tensors"]
                      and min_ratio >= 10.0)
    write_json(WORK / "logs" / "verify_merge_weights.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
