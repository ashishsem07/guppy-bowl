// Node check of web/fish-lm.js using onnxruntime-web (wasm) + @huggingface/tokenizers, the same libraries the browser loads.
import { readFileSync } from "node:fs";
import * as ort from "../jsdeps/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs";
import { Tokenizer } from "../jsdeps/node_modules/@huggingface/tokenizers/dist/tokenizers.min.mjs";
import { FishLM } from "../web/fish-lm.js";

const dir = new URL("../build/fish-lm-web/", import.meta.url);
const tokenizer = new Tokenizer(JSON.parse(readFileSync(new URL("tokenizer.json", dir))), JSON.parse(readFileSync(new URL("tokenizer_config.json", dir))));
const text = "<|im_start|>user<|im_end|>\nare you scared of the cat<|im_end|>\n<|im_start|>assistant<|im_end|>\n";
console.log("ids:", Array.from(tokenizer.encode(text, { add_special_tokens: false }).ids).join(","));
ort.env.wasm.numThreads = 1;
const lm = await FishLM.create({ ort, modelBytes: readFileSync(new URL("onnx/model_quantized.onnx", dir)), tokenizer });
const t0 = performance.now();
console.log("greedy:", await lm.reply("are you scared of the cat", { temperature: 1e-6, topK: 1 }));
console.log(`(${Math.round(performance.now() - t0)} ms)`);
for (const p of ["someone tapped the glass", "want some food", "goodnight guppy"]) console.log(">", p, "\n  ", await lm.reply(p));
