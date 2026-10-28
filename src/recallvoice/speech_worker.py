"""Isolated speech engines: JSON request on stdin, encoded audio at a supplied path."""

import asyncio
import json
import sys
from pathlib import Path


def main():
    request = json.load(sys.stdin)
    if request["provider"] == "offline":
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.save_to_file(request["text"], request["output"])
        engine.runAndWait()
    elif request["provider"] == "edge":
        import edge_tts

        asyncio.run(
            edge_tts.Communicate(request["text"], request["voice"]).save(
                request["output"]
            )
        )
    else:
        raise ValueError("Unknown speech provider")
    if Path(request["output"]).stat().st_size < 48:
        raise ValueError("Speech engine produced no usable audio")


if __name__ == "__main__":
    main()
