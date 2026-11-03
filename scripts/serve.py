"""Local static server for web/ with the same isolation headers as production."""
import http.server, functools, sys
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm"}
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(H, directory=sys.argv[2] if len(sys.argv) > 2 else "web")).serve_forever()
