from dataclasses import dataclass, field
from typing import Protocol, Sequence
import numpy as np


class ComponentError(RuntimeError):
    """An external component could not produce its contracted result."""


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class AudioClip:
    data: bytes
    mime: str
    provider: str


@dataclass(frozen=True)
class Evidence:
    source_id: int
    conversation: str
    speaker: str
    text: str
    similarity: float


@dataclass
class TurnOutcome:
    attempt: str
    recognized: str
    submitted: str
    answer: str
    sources: list[Evidence] = field(default_factory=list)
    audio: AudioClip | None = None
    notices: list[str] = field(default_factory=list)
    replayed: bool = False


class Recognizer(Protocol):
    def transcribe(self, wav: bytes) -> str: ...


class Generator(Protocol):
    def reply(self, messages: Sequence[ChatMessage]) -> str: ...


class Encoder(Protocol):
    @property
    def identity(self) -> str: ...
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class Synthesizer(Protocol):
    def render(self, text: str) -> AudioClip: ...
