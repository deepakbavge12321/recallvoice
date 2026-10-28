from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from support import Sandbox
from recallvoice.contracts import ChatMessage, ComponentError
from recallvoice.inference import OllamaGenerator


class TransportIntegration(unittest.TestCase):
    def test_actual_http_contract_and_failure_mapping(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                received.append((self.path, request))
                response = (
                    {"message": {"content": "An integration fixture reply."}}
                    if request["model"] == "fixture-model"
                    else {"error": "model absent"}
                )
                self.send_response(200 if request["model"] == "fixture-model" else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

        box = Sandbox()
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            settings = replace(
                box.settings,
                ollama_url=f"http://127.0.0.1:{server.server_port}",
                chat_model="fixture-model",
            )
            adapter = OllamaGenerator(settings)
            self.assertEqual(
                adapter.reply([ChatMessage("user", "transport probe")]),
                "An integration fixture reply.",
            )
            self.assertEqual(received[0][0], "/api/chat")
            self.assertFalse(received[0][1]["stream"])
            with self.assertRaises(ComponentError):
                OllamaGenerator(replace(settings, chat_model="absent")).reply([])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            box.close()
