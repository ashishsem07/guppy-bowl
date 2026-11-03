import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { Tokenizer } from "../jsdeps/node_modules/@huggingface/tokenizers/dist/tokenizers.min.mjs";
const dir = new URL("../build/fish-lm-web/", import.meta.url);
const tok = new Tokenizer(JSON.parse(readFileSync(new URL("tokenizer.json", dir))), JSON.parse(readFileSync(new URL("tokenizer_config.json", dir))));
const users = ["are you scared of the cat", "someone tapped the glass", "what's your name?", "I LOVE you, fish!!", "  two  spaces and émoji 🐟", "hello\nnew line", "what is 12*7"];
const texts = users.map((u) => `<|im_start|>user<|im_end|>\n${u}<|im_end|>\n<|im_start|>assistant<|im_end|>\n`);
const py = JSON.parse(execFileSync(".venv/bin/python", ["-I", "-c", `
import json,sys; from tokenizers import Tokenizer
t=Tokenizer.from_file('build/fish-lm-web/tokenizer.json'); print(json.dumps([t.encode(x).ids for x in json.loads(sys.argv[1])]))`, JSON.stringify(texts)]).toString());
let ok = 0;
texts.forEach((t, i) => { const js = Array.from(tok.encode(t, { add_special_tokens: false }).ids); const same = JSON.stringify(js) === JSON.stringify(py[i]); ok += same; if (!same) console.log("MISMATCH", JSON.stringify(users[i]), "\n js", js.join(","), "\n py", py[i].join(",")); });
console.log(`${ok}/${texts.length} prompts tokenize identically`);
const sample = py[0].slice(-8); console.log("decode check:", JSON.stringify(tok.decode([476, 286, 12427, 255, 230, 3157])));
