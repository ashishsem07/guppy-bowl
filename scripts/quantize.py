"""Build the transformers.js model folder: tokenizer + config + fp32/int8/int4 ONNX weights,
then sample a few replies from each so quality loss is visible."""
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
import onnx
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "build" / "onnx-fp32"
HF = ROOT / "build" / "fish-lm-llama"
OUT = ROOT / "build" / "fish-lm-web"
ONNX_DIR = OUT / "onnx"

if OUT.exists():
    shutil.rmtree(OUT)
ONNX_DIR.mkdir(parents=True)
for f in ["config.json", "generation_config.json"]:
    shutil.copy(SRC / f, OUT / f)
for f in ["tokenizer.json", "tokenizer_config.json"]:
    shutil.copy(HF / f, OUT / f)
shutil.copy(SRC / "model.onnx", ONNX_DIR / "model.onnx")

quantize_dynamic(str(SRC / "model.onnx"), str(ONNX_DIR / "model_quantized.onnx"), weight_type=QuantType.QUInt8)

q4 = MatMulNBitsQuantizer(onnx.load(str(SRC / "model.onnx")), block_size=32, is_symmetric=True)
q4.process()
q4.model.save_model_to_file(str(ONNX_DIR / "model_q4.onnx"), use_external_data_format=False)

for p in sorted(ONNX_DIR.iterdir()):
    print(f"{p.name:24s} {p.stat().st_size / 1e6:7.1f} MB")

# ---- quick quality check: greedy replies from each variant ----
tok = AutoTokenizer.from_pretrained(OUT)
L, KV, HD = 12, 2, 64


def reply(sess, user, max_new=40, rep=1.3):
    text = f"<|im_start|>user<|im_end|>\n{user}<|im_end|>\n<|im_start|>assistant<|im_end|>\n"
    ids = tok.encode(text)
    past = {f"past_key_values.{i}.{kv}": np.zeros((1, KV, 0, HD), np.float32) for i in range(L) for kv in ("key", "value")}
    seq, out, step_ids = list(ids), [], ids
    for _ in range(max_new):
        n = len(seq)
        feeds = {
            "input_ids": np.array([step_ids], np.int64),
            "attention_mask": np.ones((1, n), np.int64),
            "position_ids": np.arange(n - len(step_ids), n, dtype=np.int64)[None],
            **past,
        }
        feeds = {k: v for k, v in feeds.items() if k in {i.name for i in sess.get_inputs()}}
        res = sess.run(None, feeds)
        logits = res[0][0, -1].copy()
        for t in set(seq[-64:]):
            logits[t] = logits[t] / rep if logits[t] > 0 else logits[t] * rep
        nxt = int(logits.argmax())
        if nxt in (8, 3):
            break
        out.append(nxt)
        seq.append(nxt)
        step_ids = [nxt]
        names = [o.name for o in sess.get_outputs()][1:]
        past = {n_.replace("present", "past_key_values"): v for n_, v in zip(names, res[1:])}
    return tok.decode(out, skip_special_tokens=True).strip()


prompts = ["are you scared of the cat", "someone is tapping on the glass", "food is falling from the top", "the lights went off", "hello fish"]
for name in ["model.onnx", "model_quantized.onnx", "model_q4.onnx"]:
    sess = ort.InferenceSession(str(ONNX_DIR / name), providers=["CPUExecutionProvider"])
    print(f"\n== {name}")
    for p in prompts:
        print(f"  > {p}\n    {reply(sess, p)}")
