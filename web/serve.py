#!/usr/bin/env python3
import http.server
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

class COOPCOEPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "application/javascript",
        ".data": "application/octet-stream",
    }

print(f"Serving on http://localhost:{PORT}")
print("Press Ctrl+C to stop")
http.server.HTTPServer(("", PORT), COOPCOEPHandler).serve_forever()
