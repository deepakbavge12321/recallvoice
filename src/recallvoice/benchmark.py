"""Fixed source-retrieval evaluation; only real SentenceEncoder results are reported."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import tempfile
import time
from datetime import datetime, timezone
from .configuration import load_settings
from .contracts import ComponentError
from .repository import Repository
from .encoding import SentenceEncoder
from .semantic import SemanticMemory
from .vocabulary import Vocabulary


def evaluate(settings, dataset: Path):
    corpus = json.loads(dataset.read_text(encoding="utf-8"))
    report = {
        "kind": "real-model-source-benchmark",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "corpus_version": corpus["version"],
        "python": platform.python_version(),
        "dataset_sha256": hashlib.sha256(
            json.dumps(corpus, sort_keys=True).encode()
        ).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "sentence-transformers", "torch")
        },
        "query_count": len(corpus["facts"]),
        "threshold": -1.0,
        "results": [],
    }
    settings.database.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=settings.database.parent) as directory:
        store = Repository(Path(directory) / "benchmark.sqlite3")
        vocabulary = Vocabulary(store)
        vocabulary.save_rule("cedar", "Seydra")
        correct = [
            vocabulary.correct(case["before"]) == case["after"]
            for case in corpus["alias_cases"]
        ]
        report["alias_accuracy"] = sum(correct) / len(correct)
        report["alias_cases"] = len(correct)
        expected = []
        for index, fact in enumerate(corpus["facts"]):
            conversation = store.new_conversation("Fixture " + fact["label"])
            attempt = "fixture-" + str(index)
            store.begin(attempt, conversation, str(index))
            # A fixed acknowledgement is stored separately and never treated as ground truth.
            store.commit_dialogue(
                attempt,
                fact["text"]
                if fact["speaker"] == "user"
                else "Please record this recommendation.",
                fact["text"]
                if fact["speaker"] == "assistant"
                else "Recorded for this test corpus.",
            )
            expected.append(
                next(
                    row["id"]
                    for row in store.dialogue(conversation)
                    if row["speaker"] == fact["speaker"]
                )
            )
        encoder = SentenceEncoder(settings)
        memory = SemanticMemory(store, encoder)
        report["encoder"] = encoder.identity
        try:
            started = time.perf_counter()
            report["indexed_sources"] = memory.synchronize()
            report["cold_index_ms"] = (time.perf_counter() - started) * 1000
            latency = []
            for fact, target in zip(corpus["facts"], expected):
                started = time.perf_counter()
                hits = memory.search(fact["question"], 5, -1)
                latency.append((time.perf_counter() - started) * 1000)
                ids = [hit.source_id for hit in hits]
                report["results"].append(
                    {
                        "label": fact["label"],
                        "question": fact["question"],
                        "expected": target,
                        "ranked_ids": ids,
                        "scores": [hit.similarity for hit in hits],
                    }
                )
            report["recall_at_1"] = statistics.mean(
                row["expected"] in row["ranked_ids"][:1] for row in report["results"]
            )
            report["recall_at_5"] = statistics.mean(
                row["expected"] in row["ranked_ids"][:5] for row in report["results"]
            )
            report["retrieval_mean_ms"] = statistics.mean(latency)
            report["retrieval_median_ms"] = statistics.median(latency)
            report["retrieval_samples_ms"] = latency
            fresh = store.new_conversation("Question-only session")
            visible_ids = {row["id"] for row in store.dialogue(fresh)}
            report["current_session_only_recall"] = sum(
                target in visible_ids for target in expected
            ) / len(expected)
            report["status"] = "completed"
        except ComponentError as error:
            report.update(status="unavailable", error=str(error))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/corpus.json"))
    parser.add_argument("--output", type=Path, default=Path("results/semantic.json"))
    args = parser.parse_args()
    result = evaluate(load_settings(args.config), args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
