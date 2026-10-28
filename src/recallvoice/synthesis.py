import json
from pathlib import Path
import subprocess
import sys
import tempfile
from .configuration import Settings
from .contracts import AudioClip, ComponentError


class ProcessSpeech:
    def __init__(self, provider, voice, temporary_root: Path, timeout=25):
        self.provider, self.voice = provider, voice
        self.root, self.timeout = temporary_root, timeout

    def render(self, text):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=self.root) as directory:
                extension, mime = (
                    (".wav", "audio/wav")
                    if self.provider == "offline"
                    else (".mp3", "audio/mpeg")
                )
                output = Path(directory) / ("speech" + extension)
                request = {
                    "provider": self.provider,
                    "voice": self.voice,
                    "text": text,
                    "output": str(output),
                }
                child = subprocess.run(
                    [sys.executable, "-m", "recallvoice.speech_worker"],
                    input=json.dumps(request),
                    text=True,
                    capture_output=True,
                    timeout=self.timeout,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if child.returncode != 0:
                    raise RuntimeError(
                        child.stderr.strip().splitlines()[-1]
                        if child.stderr.strip()
                        else f"Worker exited {child.returncode}"
                    )
                audio = output.read_bytes()
                if len(audio) < 48:
                    raise RuntimeError("Speech output is empty")
                return AudioClip(audio, mime, self.provider)
        except Exception as error:
            raise ComponentError(f"{self.provider} speech failed: {error}") from error


class SpeechFallback:
    def __init__(self, providers):
        self.providers = providers
        self.failures = []

    def render(self, text):
        self.failures = []
        if not text.strip():
            raise ComponentError("Cannot synthesize empty text")
        for provider in self.providers:
            try:
                return provider.render(text)
            except ComponentError as error:
                self.failures.append(str(error))
        raise ComponentError(
            "; ".join(self.failures) or "No speech providers configured"
        )


def speech_service(settings: Settings):
    return SpeechFallback(
        [
            ProcessSpeech(
                name, settings.edge_voice, settings.database.parent / "speech-tmp"
            )
            for name in settings.speech_providers
        ]
    )
