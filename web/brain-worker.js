// Runs Fish LM off the main thread so the bowl keeps animating while the fish thinks.
// Messages in:  {type: "load"} | {type: "reply", id, text}
// Messages out: {type: "progress", loaded, total} | {type: "ready", ms, cached, threads}
//               {type: "reply", id, text, ms} | {type: "error", id?, message}
import * as ort from "./vendor/ort.wasm.min.mjs";
import { Tokenizer } from "./vendor/tokenizers.min.mjs";
import { FishLM } from "./fish-lm.js";

const MODEL_DIR = new URL("./model/", import.meta.url);
ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let lm = null;
let queue = Promise.resolve();

async function fetchModel(manifest) {
  const cacheName = `fish-lm-${manifest.version}`;
  let cache = null;
  try {
    cache = await caches.open(cacheName);
    for (const key of await caches.keys()) if (key.startsWith("fish-lm-") && key !== cacheName) await caches.delete(key);
  } catch { cache = null; }

  const bytes = new Uint8Array(manifest.bytes);
  let offset = 0, cached = true;
  for (const part of manifest.parts) {
    const url = new URL(part, MODEL_DIR).href;
    let res = cache ? await cache.match(url) : null;
    if (!res) {
      cached = false;
      res = await fetch(url);
      if (!res.ok) throw new Error(`Could not download ${part} (HTTP ${res.status})`);
      if (cache) { try { await cache.put(url, res.clone()); } catch { /* storage full: keep going without caching */ } }
    }
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes.set(value, offset);
      offset += value.length;
      postMessage({ type: "progress", loaded: offset, total: manifest.bytes });
    }
  }
  if (offset !== manifest.bytes) throw new Error("Model download was incomplete");
  return { bytes, cached };
}

async function load() {
  const t0 = performance.now();
  const [manifest, tokJson, tokCfg] = await Promise.all(
    ["manifest.json", "tokenizer.json", "tokenizer_config.json"].map((f) => fetch(new URL(f, MODEL_DIR)).then((r) => r.json())),
  );
  const { bytes, cached } = await fetchModel(manifest);
  lm = await FishLM.create({ ort, modelBytes: bytes, tokenizer: new Tokenizer(tokJson, tokCfg) });
  await lm.reply("hello", { maxNew: 2 }); // warm-up
  postMessage({ type: "ready", ms: Math.round(performance.now() - t0), cached, threads: ort.env.wasm.numThreads });
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "load") {
    load().catch((err) => postMessage({ type: "error", message: String(err?.message || err) }));
  } else if (msg.type === "reply") {
    queue = queue.then(async () => {
      if (!lm) return postMessage({ type: "error", id: msg.id, message: "Model is not loaded yet" });
      const t0 = performance.now();
      try {
        const text = await lm.reply(msg.text);
        postMessage({ type: "reply", id: msg.id, text, ms: Math.round(performance.now() - t0) });
      } catch (err) {
        postMessage({ type: "error", id: msg.id, message: String(err?.message || err) });
      }
    });
  }
};
