from dataclasses import dataclass, fields
from pathlib import Path
import os
import tomllib
from .contracts import ConfigurationError


@dataclass(frozen=True)
class Settings:
    database: Path = Path(".local/recallvoice.sqlite3")
    model_dir: Path = Path(".local/models")
    ollama_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen2.5:1.5b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = ""
    whisper_size: str = "base"
    whisper_device: str = "cpu"
    speech_providers: tuple[str, ...] = ("offline", "edge")
    edge_voice: str = "en-GB-SoniaNeural"
    retrieval_count: int = 5
    similarity_floor: float = 0.2
    evidence_characters: int = 3000
    request_timeout: float = 180

    def __post_init__(self):
        if not self.chat_model.strip() or not self.ollama_url.startswith(
            ("http://", "https://")
        ):
            raise ConfigurationError(
                "A model name and HTTP Ollama endpoint are required"
            )
        if self.retrieval_count < 1 or not -1 <= self.similarity_floor <= 1:
            raise ConfigurationError("Invalid retrieval settings")
        if self.evidence_characters < 200 or self.request_timeout <= 0:
            raise ConfigurationError("Context budget or timeout is too small")
        if not self.speech_providers or set(self.speech_providers) - {
            "offline",
            "edge",
        }:
            raise ConfigurationError("Speech providers must be offline and/or edge")
        if self.whisper_device not in {"cpu", "cuda"}:
            raise ConfigurationError("Whisper device must be cpu or cuda")


def load_settings(path: str | Path | None = None) -> Settings:
    chosen = Path(path or os.environ.get("RV_CONFIG", "settings.toml")).resolve()
    data = tomllib.loads(chosen.read_text(encoding="utf-8")) if chosen.exists() else {}
    known = {f.name for f in fields(Settings)}
    if set(data) - known:
        raise ConfigurationError(f"Unknown settings: {sorted(set(data) - known)}")
    defaults = Settings()
    result = {}
    for name in known:
        raw = os.environ.get(
            "RV_" + name.upper(), data.get(name, getattr(defaults, name))
        )
        try:
            if name in {"database", "model_dir"}:
                value = Path(raw).expanduser()
                raw = (
                    (chosen.parent / value).resolve()
                    if not value.is_absolute()
                    else value.resolve()
                )
            elif name in {"retrieval_count", "evidence_characters"}:
                raw = int(raw)
            elif name in {"similarity_floor", "request_timeout"}:
                raw = float(raw)
            elif name == "speech_providers":
                if not isinstance(raw, (str, list, tuple)):
                    raise ValueError(
                        "speech_providers must be an array or comma-separated text"
                    )
                raw = (
                    tuple(v.strip() for v in raw.split(","))
                    if isinstance(raw, str)
                    else tuple(raw)
                )
            elif not isinstance(raw, str):
                raise ValueError(f"{name} must be text")
        except (ValueError, TypeError) as error:
            raise ConfigurationError(f"Invalid {name}: {error}") from error
        result[name] = raw
    return Settings(**result)
