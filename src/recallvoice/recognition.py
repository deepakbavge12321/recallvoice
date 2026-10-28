import io
import wave
from .configuration import Settings
from .contracts import ComponentError


class WhisperRecognizer:
    def __init__(self, settings: Settings):
        self.settings, self._decoder = settings, None

    def transcribe(self, wav: bytes) -> str:
        try:
            with wave.open(io.BytesIO(wav)) as source:
                if source.getnframes() == 0 or source.getcomptype() != "NONE":
                    raise ValueError("Expected nonempty PCM WAV audio")
            if self._decoder is None:
                from faster_whisper import WhisperModel

                directory = self.settings.model_dir / "whisper"
                model = (
                    str(directory) if directory.exists() else self.settings.whisper_size
                )
                self._decoder = WhisperModel(
                    model,
                    device=self.settings.whisper_device,
                    compute_type="int8"
                    if self.settings.whisper_device == "cpu"
                    else "float16",
                    local_files_only=True,
                )
            segments, _ = self._decoder.transcribe(
                io.BytesIO(wav), language="en", beam_size=3, vad_filter=True
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            if not transcript:
                raise ValueError("No speech was recognized")
            return transcript
        except Exception as error:
            raise ComponentError(
                f"Whisper could not transcribe WAV input: {error}"
            ) from error
