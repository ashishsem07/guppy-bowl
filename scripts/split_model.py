"""Split the int8 ONNX model into <20 MB parts (Cloudflare Pages caps files at 25 MiB) into web/weights/<version>/ and write web/model/manifest.json.
The versioned folder is served with an immutable cache header; the manifest is not."""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
src = ROOT / "build" / "fish-lm-web" / "onnx" / "model_quantized.onnx"
out = ROOT / "web" / "model"
import shutil
data = src.read_bytes()
version = hashlib.sha256(data).hexdigest()[:12]
weights = ROOT / "web" / "weights"
shutil.rmtree(weights, ignore_errors=True)
(weights / version).mkdir(parents=True)
size = 19 * 1024 * 1024
parts = []
for i in range(0, len(data), size):
    name = f"part-{i // size:03d}.bin"
    (weights / version / name).write_bytes(data[i:i + size])
    parts.append(f"../weights/{version}/{name}")
(out / "manifest.json").write_text(json.dumps({"version": version, "bytes": len(data), "parts": parts}, indent=2))
print(version, len(data), parts)
