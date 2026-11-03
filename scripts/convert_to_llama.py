"""Convert the transformer-toolkit Fish LM checkpoint into a Hugging Face Llama model,
then check the converted model gives the same logits as the original code."""
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "downloads" / "weights"
OUT = ROOT / "build" / "fish-lm-llama"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ttk_ref.model import Transformer, TransformerConfig  # noqa: E402

cfg = json.loads((SRC / "config.json").read_text())
sd = torch.load(SRC / "model.pt", map_location="cpu", weights_only=True)
sd = sd.get("model", sd)

# ---- original model ----
ref = Transformer(TransformerConfig(
    vocab_size=cfg["vocab_size"], dim=cfg["dim"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
    n_kv_heads=cfg["n_kv_heads"], ffn="swiglu", hidden_dim=cfg["hidden_dim"], norm="rmsnorm",
    pos_enc="rope", dropout=0.0, tie_weights=False, max_seq=cfg["max_seq"],
))
ref.load_state_dict(sd)
ref.eval()

# ---- llama model ----
head_dim = cfg["dim"] // cfg["n_heads"]
kv_rows = cfg["n_kv_heads"] * head_dim
lcfg = LlamaConfig(
    vocab_size=cfg["vocab_size"], hidden_size=cfg["dim"], intermediate_size=cfg["hidden_dim"],
    num_hidden_layers=cfg["n_layers"], num_attention_heads=cfg["n_heads"], num_key_value_heads=cfg["n_kv_heads"],
    head_dim=head_dim, max_position_embeddings=cfg["max_seq"], rms_norm_eps=cfg["eps"], rope_theta=10000.0,
    hidden_act="silu", tie_word_embeddings=False, attention_bias=False, mlp_bias=False,
    bos_token_id=2, eos_token_id=[8, 3], pad_token_id=1,
)
new = {"model.embed_tokens.weight": sd["embed.weight"], "model.norm.weight": sd["norm.w"], "lm_head.weight": sd["head.weight"]}
for i in range(cfg["n_layers"]):
    p, q = f"blocks.{i}.", f"model.layers.{i}."
    kv = sd[p + "attn.kv.weight"]
    new.update({
        q + "input_layernorm.weight": sd[p + "norm1.w"],
        q + "post_attention_layernorm.weight": sd[p + "norm2.w"],
        q + "self_attn.q_proj.weight": sd[p + "attn.q.weight"],
        q + "self_attn.k_proj.weight": kv[:kv_rows],
        q + "self_attn.v_proj.weight": kv[kv_rows:],
        q + "self_attn.o_proj.weight": sd[p + "attn.out.weight"],
        q + "mlp.gate_proj.weight": sd[p + "ffn.w1.weight"],
        q + "mlp.up_proj.weight": sd[p + "ffn.w3.weight"],
        q + "mlp.down_proj.weight": sd[p + "ffn.w2.weight"],
    })
llama = LlamaForCausalLM(lcfg)
missing, unexpected = llama.load_state_dict(new, strict=False)
assert not unexpected, unexpected
assert all("rotary" in m for m in missing), missing
llama.eval()

# ---- tokenizer ----
tok = PreTrainedTokenizerFast(
    tokenizer_file=str(SRC / "tokenizer.json"),
    bos_token="[BOS]", eos_token="<|im_end|>", unk_token="[UNK]", pad_token="[PAD]",
)

# ---- parity check ----
prompt = "<|im_start|>user<|im_end|>\nare you scared of the cat<|im_end|>\n<|im_start|>assistant<|im_end|>\n"
ids = torch.tensor([tok.encode(prompt)])
with torch.no_grad():
    a = ref(ids)[0]
    b = llama(ids).logits
diff = (a - b).abs().max().item()
print(f"tokens={ids.shape[1]} max |logit diff|={diff:.2e}  argmax equal={bool((a.argmax(-1) == b.argmax(-1)).all())}")
assert diff < 1e-3, "converted model does not match the original"

with torch.no_grad():
    g = llama.generate(ids, max_new_tokens=40, do_sample=False, repetition_penalty=1.3, eos_token_id=[8, 3])
print("greedy reply:", tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))

if OUT.exists():
    shutil.rmtree(OUT)
llama.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
llama.generation_config.save_pretrained(OUT)
print("saved", OUT)
