"""Explicit public model downloads. Application adapters never download implicitly."""

import argparse
import os
from .configuration import load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    settings = load_settings(args.config)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "90")
    # A dedicated cache avoids implicit credentials from another Hugging Face login.
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download

    print("Preparing sentence model...", flush=True)
    SentenceTransformer(
        settings.embedding_model,
        revision=settings.embedding_revision or None,
        token=False,
        device="cpu",
    ).save(str(settings.model_dir / "sentence"))
    print("Preparing Whisper...", flush=True)
    snapshot_download(
        "Systran/faster-whisper-" + settings.whisper_size,
        token=False,
        local_dir=settings.model_dir / "whisper",
        max_workers=2,
        allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"],
    )
    print(f"Models ready at {settings.model_dir}", flush=True)


if __name__ == "__main__":
    main()
