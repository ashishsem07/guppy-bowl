"""Sample replies for candidate bowl-event prompts using the notebook's settings (temp 0.4, top-k 50, rep 1.3)."""
import sys
from pathlib import Path
import numpy as np, onnxruntime as ort
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parent.parent / "build" / "fish-lm-web"
tok = AutoTokenizer.from_pretrained(ROOT)
sess = ort.InferenceSession(str(ROOT / "onnx" / "model_quantized.onnx"), providers=["CPUExecutionProvider"])
outs = [o.name for o in sess.get_outputs()][1:]
rng = np.random.default_rng(0)
def reply(user, temp=0.4, top_k=50, rep=1.3, max_new=40):
    seq = tok.encode(f"<|im_start|>user<|im_end|>\n{user}<|im_end|>\n<|im_start|>assistant<|im_end|>\n")
    past = {f"past_key_values.{i}.{kv}": np.zeros((1, 2, 0, 64), np.float32) for i in range(12) for kv in ("key", "value")}
    step, out = list(seq), []
    for _ in range(max_new):
        n = len(seq)
        res = sess.run(None, {"input_ids": np.array([step], np.int64), "attention_mask": np.ones((1, n), np.int64),
                              "position_ids": np.arange(n - len(step), n, dtype=np.int64)[None], **past})
        lg = res[0][0, -1] / temp
        for t in set(seq[-64:]): lg[t] /= rep
        kth = np.sort(lg)[-top_k]; lg = np.where(lg < kth, -np.inf, lg)
        p = np.exp(lg - lg.max()); p /= p.sum()
        nxt = int(rng.choice(len(p), p=p))
        seq.append(nxt)
        if nxt in (8, 3): break
        out.append(nxt); step = [nxt]
        past = {k.replace("present", "past_key_values"): v for k, v in zip(outs, res[1:])}
    return tok.decode(out, skip_special_tokens=True).strip()
for p in sys.argv[1:]:
    print(f"> {p}")
    for _ in range(3): print("   ", reply(p))
