from collections.abc import Sequence
import requests
from .configuration import Settings
from .contracts import ChatMessage, ComponentError


class OllamaGenerator:
    def __init__(self, settings: Settings, transport=None):
        self.settings = settings
        self.transport = transport or requests.Session()

    def reply(self, messages: Sequence[ChatMessage]) -> str:
        body = {
            "model": self.settings.chat_model,
            "stream": False,
            "messages": [
                {"role": item.role, "content": item.content} for item in messages
            ],
            "options": {
                "temperature": 0,
                "seed": 17,
                "num_ctx": 2048,
                "num_predict": 180,
            },
        }
        try:
            response = self.transport.post(
                self.settings.ollama_url.rstrip("/") + "/api/chat",
                json=body,
                timeout=(5, self.settings.request_timeout),
            )
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}: {response.text[:300]}")
            answer = response.json()["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Empty or non-text answer")
            return answer.strip()
        except Exception as error:
            raise ComponentError(
                f"Ollama generation failed for {self.settings.chat_model}: {error}"
            ) from error
