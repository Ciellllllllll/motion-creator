"""LlamaBiModel が実際に双方向注意で動くかを、小さなランダムモデルで実測する（重みの取得は不要）。

transformers 5.1 の LlamaModel.forward は create_causal_mask を使い、LlamaBiModel が上書きする
_update_causal_mask を呼ばない。そのため「パディングなし・SDPA でマスクが省略され、
is_causal=False が効く」場合だけ双方向になると考えられる。Kimodo は1文ずつ（パディングなし）で
エンコードするので、その条件を含めて実測する。

判定：先頭トークンの出力が末尾トークンの変更で変われば双方向、変わらなければ一方向。
"""
import json
import sys

import torch
from transformers import LlamaConfig

from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel


def first_token_change(model, ids, mask, device):
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 7) % model.config.vocab_size
    with torch.no_grad():
        a = model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
        b = model(input_ids=ids2.to(device), attention_mask=mask.to(device)).last_hidden_state
    # 左パディングのため、有効な先頭トークンの位置を使う
    first = int((mask[0] == 0).sum())
    return float((a[0, first] - b[0, first]).abs().max())


def main():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    results = {}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for impl in ["sdpa", "eager"]:
            cfg._attn_implementation = impl
            model = LlamaBiModel(cfg).to(device).eval()
            ids = torch.randint(0, 128, (1, 10))
            nopad = first_token_change(model, ids, torch.ones_like(ids), device)
            # バッチ内で長さが違うときの左パディング（Kimodo の通常経路では起きない）
            pad_mask = torch.ones_like(ids)
            pad_mask[0, :3] = 0
            padded = first_token_change(model, ids, pad_mask, device)
            results[f"{device}/{impl}"] = {
                "no_padding_first_token_delta": nopad, "bidirectional_no_padding": nopad > 1e-6,
                "left_padding_first_token_delta": padded, "bidirectional_left_padding": padded > 1e-6,
            }
    print(json.dumps(results, indent=1))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
