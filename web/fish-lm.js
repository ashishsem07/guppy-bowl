// Fish LM (68M, Govind222/68m-base-guppy-sft) running on ONNX Runtime.
// Same chat template and sampling as the author's Fish_LM.ipynb chat_sft():
// temperature 0.4, top-k 50, repetition penalty 1.3 over the last 64 tokens, stop on <|im_end|> or [EOS].

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
    return `<|im_start|>user<|im_end|>\n${userText}<|im_end|>\n<|im_start|>assistant<|im_end|>\n`;
  }

  async reply(userText, { temperature = 0.4, topK = 50, repPenalty = 1.3, maxNew = 40, random = Math.random } = {}) {
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
      for (const t of new Set(seq.slice(-64))) logits[t] /= repPenalty;

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

function sampleTopK(logits, k, random) {
  const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, k);
  const max = logits[idx[0]];
  const w = idx.map((i) => Math.exp(logits[i] - max));
  let r = random() * w.reduce((a, b) => a + b, 0);
  for (let j = 0; j < idx.length; j++) {
    r -= w[j];
    if (r <= 0) return idx[j];
  }
  return idx[idx.length - 1];
}
