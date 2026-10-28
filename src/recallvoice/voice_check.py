"""Fresh synthetic-audio end-to-end check against real local model adapters."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import tempfile
import uuid
import numpy as np
import requests
from .configuration import load_settings
from .contracts import AudioClip
from .factory import build
from .synthesis import ProcessSpeech

UTTERANCES = (
    "For the project Cedar, please remember that our verification phrase is velvet compass.",
    "What is the verification phrase for Cedar?",
)


def audio_evidence(clip: AudioClip):
    import av

    frames = 0
    seconds = 0.0
    nonzero = False
    with av.open(io.BytesIO(clip.data)) as container:
        for frame in container.decode(audio=0):
            frames += 1
            seconds += frame.samples / frame.sample_rate
            nonzero = nonzero or bool(np.any(frame.to_ndarray() != 0))
    return {
        "provider": clip.provider,
        "mime": clip.mime,
        "bytes": len(clip.data),
        "frames": frames,
        "seconds": seconds,
        "non_silent": nonzero,
        "sha256": hashlib.sha256(clip.data).hexdigest(),
    }


def verify(settings, output):
    report = {
        "kind": "real-adapter-voice-check",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "input_source": "Synthetic OS speech, not a microphone/human recording",
        "status": "running",
        "model": settings.chat_model,
        "turns": [],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("faster-whisper", "av", "sentence-transformers", "pyttsx3")
        },
        "checks": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    settings.database.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(
            settings.ollama_url.rstrip("/") + "/api/tags", timeout=10
        )
        response.raise_for_status()
        report["installed_models"] = response.json()
        with tempfile.TemporaryDirectory(dir=settings.database.parent) as directory:
            isolated = replace(
                settings, database=Path(directory) / "voice-check.sqlite3"
            )
            agent = build(isolated)
            agent.vocabulary.save_rule("cedar", "Seydra")
            agent.vocabulary.save_pronunciation("Seydra", "SEE dra")
            report["recognition_rules"] = agent.vocabulary.rules()
            fixture_speech = ProcessSpeech(
                "offline", settings.edge_voice, Path(directory) / "input-audio"
            )
            report["encoder"] = agent.memory.encoder.identity
            ids = []
            captured = []
            for index, utterance in enumerate(UTTERANCES):
                generated = fixture_speech.render(utterance)
                captured.append(generated.data)
                owner = agent.repository.new_conversation(
                    f"Voice acceptance {index + 1}"
                )
                before = agent.repository.dialogue(owner)
                submission = uuid.uuid4().hex
                row = {
                    "conversation": owner,
                    "submission": submission,
                    "intended": utterance,
                    "input": audio_evidence(generated),
                    "history_before": before,
                }
                report["turns"].append(row)
                save()
                try:
                    outcome = agent.submit(
                        owner,
                        submission,
                        wav=generated.data,
                        speak=True,
                        pronunciation=True,
                    )
                except Exception:
                    row["observations"] = agent.repository.observations(submission)
                    save()
                    raise
                row.update(
                    {
                        "recognized": outcome.recognized,
                        "submitted": outcome.submitted,
                        "answer": outcome.answer,
                        "sources": [asdict(hit) for hit in outcome.sources],
                        "notices": outcome.notices,
                        "observations": agent.repository.observations(submission),
                        "output": audio_evidence(outcome.audio)
                        if outcome.audio
                        else None,
                    }
                )
                ids.append(owner)
                save()
                print(
                    json.dumps(
                        {
                            "turn": index + 1,
                            "recognized": outcome.recognized,
                            "submitted": outcome.submitted,
                            "answer": outcome.answer,
                            "audio_provider": outcome.audio.provider
                            if outcome.audio
                            else None,
                        }
                    ),
                    flush=True,
                )
            first, second = report["turns"]
            vectors_before = len(
                agent.repository.vectors(agent.memory.encoder.identity)
            )
            replay = agent.submit(
                ids[1], second["submission"], wav=captured[1], speak=False
            )
            report["checks"] = {
                "nonempty_recognition": all(
                    t["recognized"].strip() for t in report["turns"]
                ),
                "observed_alias_changes": all(
                    t["recognized"] != t["submitted"] and "Seydra" in t["submitted"]
                    for t in report["turns"]
                ),
                "independent_conversations": ids[0] != ids[1]
                and second["history_before"] == [],
                "prior_user_source_retrieved": any(
                    s["conversation"] == ids[0] and s["speaker"] == "user"
                    for s in second["sources"]
                ),
                "correct_phrase_recalled": "velvet compass"
                in " ".join(re.findall(r"\w+", second["answer"].lower())),
                "valid_non_silent_speech": all(
                    t["output"]
                    and t["output"]["frames"] > 0
                    and t["output"]["non_silent"]
                    for t in report["turns"]
                ),
                "completed_atomic_exchanges": all(
                    len(agent.repository.dialogue(owner)) == 2 for owner in ids
                ),
                "no_semantic_degradation": all(
                    not t["notices"] for t in report["turns"]
                ),
                "idempotent_replay": replay.replayed
                and len(agent.repository.vectors(agent.memory.encoder.identity))
                == vectors_before,
                "stage_measurements_complete": all(
                    {o["stage"] for o in t["observations"]}
                    == {
                        "recognition",
                        "retrieval",
                        "generation",
                        "persistence",
                        "indexing",
                        "speech",
                    }
                    for t in report["turns"]
                ),
            }
            report["status"] = "passed" if all(report["checks"].values()) else "failed"
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", type=Path, default=Path("results/voice.json"))
    args = parser.parse_args()
    report = verify(load_settings(args.config), args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": report["checks"],
                "error": report.get("error"),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
