import hashlib
from .configuration import Settings
from .contracts import ComponentError


class SentenceEncoder:
    def __init__(self, settings: Settings):
        self.settings, self._instance, self._identity = settings, None, None

    @property
    def identity(self):
        if self._identity is None:
            weights = self.settings.model_dir / "sentence" / "model.safetensors"
            digest = "cached-id"
            if weights.exists():
                with weights.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
            self._identity = f"{self.settings.embedding_model}@{self.settings.embedding_revision or 'unspecified'}:{digest}"
        return self._identity

    def encode(self, texts):
        try:
            if self._instance is None:
                from sentence_transformers import SentenceTransformer

                local = self.settings.model_dir / "sentence"
                location = (
                    str(local) if local.exists() else self.settings.embedding_model
                )
                self._instance = SentenceTransformer(
                    location,
                    device="cpu",
                    local_files_only=True,
                    token=False,
                    revision=self.settings.embedding_revision or None,
                )
            return self._instance.encode(
                list(texts),
                batch_size=24,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as error:
            raise ComponentError(
                f"Local sentence model unavailable; run recallvoice-prepare: {error}"
            ) from error
