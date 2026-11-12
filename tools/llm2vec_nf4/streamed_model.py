"""FP32 の LlamaBiModel を、層ごとに重みを読み込み・解放しながら扱う（少ないコミットで全精度処理）。

背景: Windows では safetensors の safe_open がファイルサイズ分のコミットを使い、WDDM では
GPU 確保分もコミットに数えられる。FP32 全体（約28GiB）を一度に置くと 64GB 機の仮想メモリが尽きる。

方式:
- モデル構造と forward は LlamaBiModel（nv-tlabs/kimodo 実装）そのもの。
- 埋め込み・RMSNorm・RoPE は常駐。各デコーダー層の Linear 7個の重みだけを、その層の forward 直前に
  safetensors から FP32 で読み込み、直後に空の置き場に戻す（forward pre/post フック）。
- PEFT の LoRA はこのモデルに通常どおり注入できる（LoRA 重みは常駐、基礎重みだけが入れ替わる）。
- 計算は GPU の FP32（TF32 無効）。全精度モデル全体を GPU に載せることはしない。
"""
import json
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from torch import nn
from transformers import LlamaConfig

from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def weight_map(d: Path) -> dict:
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
    with safe_open(d / "model.safetensors", "pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


class StreamedBiModel:
    def __init__(self, model_dir, device: str = "cuda:0", dtype: torch.dtype = torch.float32):
        self.dir = Path(model_dir)
        self.device = device
        self.dtype = dtype  # 既定は FP32（全精度）。診断で FP16 計算を測るときだけ変える
        wm = weight_map(self.dir)
        config = LlamaConfig.from_pretrained(str(self.dir))
        config._attn_implementation = "sdpa"
        with init_empty_weights(include_buffers=False):
            model = LlamaBiModel(config)
        model.config._attn_implementation = "sdpa"
        self.model = model
        # 置き場パラメーターは全層で共有するため、以後の named_parameters() は重複を省く。名前は今のうちに控える
        names = [n for n, _ in model.named_parameters()]
        self.names = list(names)
        self.src = {}  # パラメーター名 -> (シャード, 元のキー)
        self.unexpected = []
        for k, shard in wm.items():
            n = k[len("model."):] if k.startswith("model.") else k
            if n in names:
                assert n not in self.src, f"duplicate {n}"
                self.src[n] = (shard, k)
            else:
                self.unexpected.append(k)
        missing = sorted(set(names) - set(self.src))
        assert not missing, f"missing tensors: {missing[:5]} ({len(missing)})"
        self.placeholder = nn.Parameter(torch.empty(0, device=device, dtype=dtype), requires_grad=False)
        self.layer_params = defaultdict(list)  # 層番号 -> [(Linear, パラメーター名)]
        resident = []
        for n in names:
            parts = n.split(".")
            if parts[0] == "layers" and ".".join(parts[2:4]) in LINEARS and parts[-1] == "weight":
                mod = model.get_submodule(n.rsplit(".", 1)[0])
                mod._parameters["weight"] = self.placeholder
                self.layer_params[int(parts[1])].append((mod, n))
            else:
                resident.append(n)
        for shard, items in self._by_shard(resident).items():
            with safe_open(self.dir / shard, "pt") as f:
                for n, k in items:
                    set_module_tensor_to_device(model, n, device, value=f.get_tensor(k).to(device, dtype), dtype=dtype)
        model.rotary_emb.to(device)  # inv_freq は FP32 のまま（transformers の通常読み込みと同じ）
        assert len(self.layer_params) == config.num_hidden_layers
        assert all(len(v) == len(LINEARS) for v in self.layer_params.values())
        for i, layer in enumerate(model.layers):
            # フックが値を返すと入力・出力が置き換わるので、必ず None を返す
            layer.register_forward_pre_hook(lambda m, args, i=i: (self.load_layer(i), None)[1])
            layer.register_forward_hook(lambda m, args, out, i=i: (self.release_layer(i), None)[1])
        model.eval()

    def _by_shard(self, names):
        out = defaultdict(list)
        for n in names:
            shard, k = self.src[n]
            out[shard].append((n, k))
        return out

    def load_layer(self, i):
        """層 i の Linear 重みを FP32 で読み込む。読み込んだ元の dtype を返す。"""
        mods = dict((n, m) for m, n in self.layer_params[i])
        dtypes = set()
        for shard, items in self._by_shard(list(mods)).items():
            with safe_open(self.dir / shard, "pt") as f:
                for n, k in items:
                    t = f.get_tensor(k)
                    dtypes.add(str(t.dtype))
                    mods[n]._parameters["weight"] = nn.Parameter(t.to(self.device, self.dtype), requires_grad=False)
        return dtypes

    def release_layer(self, i):
        for m, _ in self.layer_params[i]:
            m._parameters["weight"] = self.placeholder
