"""公式 Llama 3 に MNTP → supervised の順で LoRA 追加重みを各1回統合し、FP16 Safetensors で保存する。

実行: python -s merge.py
- LlamaBiModel（nv-tlabs/kimodo 実装）に PEFT で両方の追加重みを注入し、層ごとに基礎重みを FP32 で
  読み込んで、PEFT の lora.Linear.merge() で MNTP → supervised の順に統合する。
- 統合済み重みは FP16 へ1回だけ丸めて、HF 形式（index つき）の Safetensors シャードに書く。
- 64GB 機の仮想メモリに収めるため streamed_model で層ごとに読み込む（全精度全体を一度に置かない）。
- trust_remote_code は使わない。オフライン（common が HF_HUB_OFFLINE=1 を設定）。
"""
import datetime as dt
import shutil
import time

import common
from common import (BASE_DIR, BASE_NAME, LLM2VEC_DEFAULTS, LORA_TARGETS, MERGED_DIR, MNTP_DIR, SUP_DIR, WORK,
                    load_json, write_json)

import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from safetensors import safe_open
from safetensors.torch import save_file

from streamed_model import StreamedBiModel

SHARD_BYTES = 5 * 1024**3
ADAPTERS = (("mntp", MNTP_DIR), ("supervised", SUP_DIR))


def check_adapter_config(d):
    c = load_json(d / "adapter_config.json")
    problems = []
    expect = {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "use_rslora": False, "fan_in_fan_out": False,
              "bias": "none", "modules_to_save": None, "base_model_name_or_path": BASE_NAME}
    for k, v in expect.items():
        if c.get(k) != v:
            problems.append(f"{k}={c.get(k)!r} (expected {v!r})")
    if set(c["target_modules"]) != set(LORA_TARGETS):
        problems.append(f"target_modules={c['target_modules']}")
    for k in ("alpha_pattern", "rank_pattern"):
        if c.get(k):
            problems.append(f"{k} not empty")
    if c.get("use_dora"):
        problems.append("use_dora")
    assert not problems, f"{d.name}: {problems}"
    return {"r": c["r"], "lora_alpha": c["lora_alpha"], "scaling": c["lora_alpha"] / c["r"],
            "target_modules": sorted(c["target_modules"])}


def check_adapter_loaded(model, name, d):
    """ファイルの全 LoRA テンソルが PEFT モデルに正しく載ったことを確認する。"""
    params = dict(model.named_parameters())
    n = 0
    with safe_open(d / "adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            pk = k.replace(".lora_A.weight", f".lora_A.{name}.weight").replace(".lora_B.weight", f".lora_B.{name}.weight")
            assert pk in params, f"{name}: adapter key not in model: {k}"
            p = params[pk]
            assert p.dtype == torch.float32, (k, p.dtype)
            assert torch.equal(p.detach().cpu(), f.get_tensor(k).to(torch.float32)), k
            n += 1
    return n


class ShardWriter:
    def __init__(self, out_dir, sizes):
        self.out_dir, self.shard, self.bytes, self.idx, self.wm, self.total = out_dir, {}, 0, 1, {}, 0
        counts = [0]
        for b in sizes:  # 書き出す順と同じ規則で、先にシャード数を決める
            if counts[-1] + b > SHARD_BYTES and counts[-1] > 0:
                counts.append(0)
            counts[-1] += b
        self.n = len(counts)

    def _name(self):
        return f"model-{self.idx:05d}-of-{self.n:05d}.safetensors"

    def add(self, name, t):
        t = t.detach().to(device="cpu", dtype=torch.float16).contiguous()
        if not torch.isfinite(t).all():
            raise RuntimeError(f"FP16 overflow/non-finite: {name}")
        b = t.numel() * 2
        if self.bytes + b > SHARD_BYTES and self.shard:
            save_file(self.shard, str(self.out_dir / self._name()), metadata={"format": "pt"})
            self.shard, self.bytes, self.idx = {}, 0, self.idx + 1
        self.shard[name] = t
        self.bytes += b
        self.total += b
        self.wm[name] = self._name()

    def close(self):
        save_file(self.shard, str(self.out_dir / self._name()), metadata={"format": "pt"})
        assert self.idx == self.n, (self.idx, self.n)
        write_json(self.out_dir / "model.safetensors.index.json",
                   {"metadata": {"total_size": self.total}, "weight_map": self.wm})
        return self.n


def main():
    log = {"started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "order": ["base", "mntp", "supervised"],
           "precision": "基礎重み(bf16)を FP32 で読み、PEFT lora.Linear.merge で MNTP→supervised の順に統合、FP16 へ1回だけ丸める",
           "placement": "streamed_model: 層ごとに GPU(FP32, TF32無効) へ読み込み・統合・書き出し・解放"}
    log["adapters"] = {k: check_adapter_config(d) for k, d in ADAPTERS}

    t0 = time.time()
    sm = StreamedBiModel(BASE_DIR)
    assert sm.unexpected in ([], ["lm_head.weight"]), sm.unexpected
    log["base_load"] = {"class": type(sm.model).__name__, "unexpected_keys": sm.unexpected,
                        "attn_implementation": sm.model.config._attn_implementation}
    order = sm.names
    streamed_names = {n: m for items in sm.layer_params.values() for m, n in items}
    shapes = {n: ((streamed_names[n].out_features, streamed_names[n].in_features) if n in streamed_names
                  else tuple(sm.model.get_parameter(n).shape)) for n in order}
    assert len(order) == 290, len(order)  # 埋め込み1 + 32層×9 + norm1

    model = PeftModel.from_pretrained(sm.model, str(MNTP_DIR), adapter_name="mntp", is_trainable=False)
    model.load_adapter(str(SUP_DIR), adapter_name="supervised", is_trainable=False)
    log["adapter_tensors_loaded_and_equal"] = {k: check_adapter_loaded(model, k, d) for k, d in ADAPTERS}
    lora_by_base = {}
    for _, m in model.named_modules():
        if isinstance(m, LoraLayer):
            lora_by_base[id(m.get_base_layer())] = m
    assert len(lora_by_base) == 32 * len(LORA_TARGETS), len(lora_by_base)

    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    MERGED_DIR.mkdir(parents=True)
    writer = ShardWriter(MERGED_DIR, [shapes[n][0] * (shapes[n][1] if len(shapes[n]) > 1 else 1) * 2 for n in order])
    resident = dict(model.base_model.model.named_parameters())
    changed = 0
    merged_count = 0
    done_layers = set()
    with torch.no_grad():
        for n in order:
            parts = n.split(".")
            if parts[0] == "layers":
                i = int(parts[1])
                streamed = {nn_: m for m, nn_ in sm.layer_params[i]}
                if n in streamed:
                    if i not in done_layers:
                        dtypes = sm.load_layer(i)
                        assert dtypes == {"torch.bfloat16"}, dtypes
                        for base_lin in streamed.values():
                            w = base_lin.weight
                            changed += int((w.to(torch.float16).to(torch.float32) != w).sum())
                            lora = lora_by_base[id(base_lin)]
                            lora.merge(adapter_names=["mntp"])
                            lora.merge(adapter_names=["supervised"])
                            assert lora.merged_adapters == ["mntp", "supervised"], lora.merged_adapters
                            merged_count += 1
                        done_layers.add(i)
                        layer_tensors = {nn_: m.weight for nn_, m in streamed.items()}
                    writer.add(n, layer_tensors[n])
                    if all(nn_ in writer.wm for nn_ in streamed):
                        sm.release_layer(i)
                    continue
            p = resident[n]  # 埋め込み・RMSNorm（PEFT の対象外なので名前は変わらない）
            assert p.device.type != "meta" and p.numel() > 0, n
            writer.add(n, p)
    log["shards"] = writer.close()
    log["lora_modules_merged"] = merged_count
    assert merged_count == 32 * len(LORA_TARGETS)
    log["base_bf16_to_fp16_changed_elements_in_lora_targets"] = changed
    log["structure"] = {"tensors": len(order), "names_shapes_equal_base": sorted(writer.wm) == sorted(order),
                        "all_finite_fp16": True}
    assert log["structure"]["names_shapes_equal_base"]
    log["merge_seconds"] = round(time.time() - t0, 1)

    cfg = sm.model.config
    cfg.architectures = ["LlamaBiModel"]
    cfg.dtype = torch.float16
    cfg.save_pretrained(str(MERGED_DIR))
    # トークナイザーは公式ファイルをそのままコピー（バイト一致で追跡できる）
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copyfile(BASE_DIR / name, MERGED_DIR / name)
    l2v = dict(LLM2VEC_DEFAULTS)
    src_cfg = SUP_DIR / "llm2vec_config.json"
    if src_cfg.exists():
        l2v.update(load_json(src_cfg))
        log["llm2vec_config_source"] = str(src_cfg)
    else:
        log["llm2vec_config_source"] = "LLM2Vec 既定値（supervised リポジトリに llm2vec_config.json なし）"
    write_json(MERGED_DIR / "llm2vec_config.json", l2v)

    c = load_json(MERGED_DIR / "config.json")
    c["_name_or_path"] = BASE_NAME  # Kimodo のプロンプト整形はこの名前で分岐する
    c.pop("quantization_config", None)
    write_json(MERGED_DIR / "config.json", c)
    log["saved_config"] = {k: c.get(k) for k in ("_name_or_path", "architectures", "dtype", "torch_dtype")}
    log["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": common.sha256(p)}
                    for p in sorted(MERGED_DIR.iterdir()) if p.is_file()}
    log["finished_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(WORK / "logs" / "merge_log.json", log)
    print("merged ->", MERGED_DIR)


if __name__ == "__main__":
    main()
