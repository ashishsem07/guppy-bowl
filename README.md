# Guppy Bowl

A digital fish bowl for a fish language model. Tap the glass, drop food, turn the lights off or talk to Guppy, and a 68M-parameter language model, running entirely in your browser, decides what it says. The bowl turns that into how Guppy feels and what it does next.

**Live:** https://guppy-bowl.vercel.app

The model is [Fish LM 68M](https://huggingface.co/Govind222/68m-base-guppy-sft) by Govind Barbade, pretrained from scratch with [transformer-toolkit](https://github.com/Barbade22/transformer-toolkit) and fine-tuned on [guppylm-60k-generic](https://huggingface.co/datasets/arman-bd/guppylm-60k-generic). This repo converts it to ONNX, quantizes it to 8-bit (70 MB, same replies as full precision) and gives it a body.

## How it works

```
tap / food / lights / chat / boredom
        │
        ├──► bowl reflex (instant): dart away, turn toward food
        │
        └──► event → short line to Guppy ──► Fish LM in a Web Worker ──► reply
                    "someone tapped the glass"                         "please don't tap. it's like
                                                                         someone cracking my whole world."
                                     │
                     mood + action ◄─┘  (from the event; when you chat, words in the reply can shift it)
```

- **Body:** `web/index.html` draws the bowl on a canvas and runs the fish's steering, reflexes, hunger, energy and trust.
- **Brain:** `web/brain-worker.js` downloads the model once (four 19 MB parts, cached with the Cache API) and runs it with ONNX Runtime Web (WASM, multi-threaded when the page is cross-origin isolated).
- **Generation:** `web/fish-lm.js` uses the model's training chat template and the same sampling as the author's notebook: temperature 0.4, top-k 50, repetition penalty 1.3.
- **Event phrasing:** Fish LM was fine-tuned on short chats, so each bowl event becomes a short line worded like its training data. See `plan()` in `web/index.html`. Until the model loads, a small rule-based stand-in brain answers.

Nothing leaves the visitor's device. There is no server.

## Run it locally

```bash
python3 scripts/serve.py 8765 web    # serves web/ with the COOP/COEP headers production uses
open http://127.0.0.1:8765
```

## Deploy your own

`web/` is a plain static site with every file under 25 MiB.

- **Vercel:** `cd web && vercel deploy --prod` (headers come from `web/vercel.json`)
- **Cloudflare Pages:** `npx wrangler pages deploy web` (headers come from `web/_headers`)
- **Anything else:** serve `web/` and send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless` for multi-threaded inference. Without them it still works on one thread.

## Rebuild the model

Only needed to change the model or quantization. The pipeline downloads the original PyTorch checkpoint, converts it to a Hugging Face Llama model and checks that its logits match the original code (max difference about 2e-5).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p downloads/weights && cd downloads/weights
for f in model.pt config.json tokenizer.json; do curl -LO https://huggingface.co/Govind222/68m-base-guppy-sft/resolve/main/$f; done
cd ../..

.venv/bin/python -I scripts/convert_to_llama.py      # -> build/fish-lm-llama, with parity check
.venv/bin/optimum-cli export onnx --model build/fish-lm-llama --task text-generation-with-past build/onnx-fp32
.venv/bin/python -I scripts/quantize.py              # -> build/fish-lm-web: fp32, int8, int4 + reply comparison
.venv/bin/python -I scripts/split_model.py           # -> web/weights/<hash>/ and web/model/manifest.json
cp build/fish-lm-web/tokenizer*.json web/model/
```

Checks for the browser runtime (`cd jsdeps && npm ci` first):

```bash
node scripts/tok_parity.mjs        # tokenizers.js ids == the Rust tokenizer used in training
node scripts/test_fish_lm.mjs      # runs web/fish-lm.js in Node on the int8 model
node scripts/browser_test.mjs http://127.0.0.1:8765/ shot   # headless Chrome end-to-end (uses local Chrome)
.venv/bin/python -I scripts/probe_events.py "someone tapped the glass" "want some food"   # try event phrasings
```

## Ideas to build on

- **Let the model choose the behaviour.** Fine-tune on tagged replies like `<mood:scared> <act:hide> i hide behind the plant` so the event table in `plan()` is no longer needed. The bowl already understands that format (`parseReply`).
- **A smaller first download.** Quantize the embedding and output layers too; they are most of the 70 MB.
- **More fish, more tank.** A second fish that the first one talks to, day and night cycles, decorations Guppy has opinions about.
- **Memory.** Keep a short log of what happened today and pick the event lines from it.

`src/bowl-prototype.html` is the first version of the bowl, with only the stand-in brain.

## License

Code in this repo: Apache-2.0 (see `LICENSE`). The model weights in `web/weights/` and the reference code in `scripts/ttk_ref/` are Govind Barbade's work under Apache-2.0; see `NOTICE` for all third-party credits.
