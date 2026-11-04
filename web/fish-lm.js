// Fish LM (68M, Govind222/68m-base-guppy-sft) running on ONNX Runtime.
// Exact chat template matching Fish LM app.py (build_chatml_prompt):
// <|im_start|>user<|im_end|>\n{clean_prompt or '...'}<|im_end|>\n<|im_start|>assistant<|im_end|>\n
// Repetition penalty matching torch.where(scores < 0, scores * rep, scores / rep) over last 64 tokens,
// strictly excluding STOP_IDS so the EOS token is never suppressed!

const N_LAYERS = 12, N_KV_HEADS = 2, HEAD_DIM = 64;
const STOP_IDS = new Set([8, 3]); // <|im_end|>, [EOS]

export class FishLM {
  constructor(ort, session, tokenizer) {
    this.ort = ort;
    this.session = session;
    this.tokenizer = tokenizer;
  }

  static async create({ ort, modelBytes, tokenizer, sessionOptions = {} }) {
    const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"], ...sessionOptions });
    return new FishLM(ort, session, tokenizer);
  }

  prompt(userText) {
    const clean = (userText || "").trim() || "...";
    return `<|im_start|>user<|im_end|>\n${clean}<|im_end|>\n<|im_start|>assistant<|im_end|>\n`;
  }

  async reply(userText, { temperature = 0.46, topK = 50, repPenalty = 1.25, maxNew = 30, random = Math.random } = {}) {
    const { ort } = this;
    const seq = Array.from(this.tokenizer.encode(this.prompt(userText), { add_special_tokens: false }).ids);
    const out = [];
    let step = seq.slice();
    let past = {};
    for (let i = 0; i < N_LAYERS; i++) {
      for (const kv of ["key", "value"]) past[`past_key_values.${i}.${kv}`] = new ort.Tensor("float32", new Float32Array(0), [1, N_KV_HEADS, 0, HEAD_DIM]);
    }

    for (let n = 0; n < maxNew; n++) {
      const total = seq.length, start = total - step.length;
      const feeds = {
        input_ids: new ort.Tensor("int64", BigInt64Array.from(step, BigInt), [1, step.length]),
        attention_mask: new ort.Tensor("int64", new BigInt64Array(total).fill(1n), [1, total]),
        position_ids: new ort.Tensor("int64", BigInt64Array.from({ length: step.length }, (_, j) => BigInt(start + j)), [1, step.length]),
        ...past,
      };
      const res = await this.session.run(feeds);
      const vocab = res.logits.dims[2];
      const all = res.logits.data;
      const logits = new Float32Array(vocab);
      const off = (step.length - 1) * vocab;
      for (let v = 0; v < vocab; v++) logits[v] = all[off + v] / Math.max(temperature, 1e-8);
      
      // Repetition penalty on recent context tokens - exclude STOP_IDS so EOS is not suppressed
      for (const t of new Set(seq.slice(-64))) {
        if (STOP_IDS.has(t) || t === 7) continue;
        logits[t] = logits[t] > 0 ? logits[t] / repPenalty : logits[t] * repPenalty;
      }

      const next = sampleTopK(logits, topK, random);
      seq.push(next);
      for (const name of Object.keys(past)) past[name].dispose?.();
      res.logits.dispose?.();
      if (STOP_IDS.has(next)) break;
      out.push(next);
      step = [next];
      past = {};
      for (let i = 0; i < N_LAYERS; i++) {
        for (const kv of ["key", "value"]) past[`past_key_values.${i}.${kv}`] = res[`present.${i}.${kv}`];
      }
    }
    return this.tokenizer.decode(out, { skip_special_tokens: true }).trim();
  }
}

function sampleTopK(logits, topK, random) {
  const n = logits.length;
  const k = Math.min(topK, n);
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => logits[b] - logits[a]);
  const kth = logits[idx[k - 1]];

  let maxL = -Infinity;
  for (let i = 0; i < k; i++) if (logits[idx[i]] > maxL) maxL = logits[idx[i]];

  let sum = 0;
  const probs = new Float32Array(k);
  for (let i = 0; i < k; i++) {
    const p = Math.exp(logits[idx[i]] - maxL);
    probs[i] = p;
    sum += p;
  }

  let r = random() * sum;
  for (let i = 0; i < k; i++) {
    r -= probs[i];
    if (r <= 0) return idx[i];
  }
  return idx[k - 1];
}
