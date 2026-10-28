import json
import numpy as np
from .contracts import ComponentError, Encoder, Evidence
from .repository import Repository


class SemanticMemory:
    def __init__(self, repository: Repository, encoder: Encoder):
        self.repository, self.encoder = repository, encoder

    def _encode(self, texts):
        try:
            matrix = np.asarray(self.encoder.encode(texts), dtype=np.float32)
            if (
                matrix.ndim != 2
                or matrix.shape[0] != len(texts)
                or matrix.shape[1] == 0
                or not np.isfinite(matrix).all()
            ):
                raise ValueError("Encoder returned an invalid matrix")
            magnitude = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(magnitude <= 0):
                raise ValueError("Encoder returned a zero-length vector")
            return matrix / magnitude
        except Exception as error:
            raise ComponentError(f"Semantic encoding unavailable: {error}") from error

    def synchronize(self, rebuild=False):
        identity = self.encoder.identity
        known = (
            set()
            if rebuild
            else {row["source"] for row in self.repository.vectors(identity)}
        )
        candidates = [
            row for row in self.repository.sources() if row["id"] not in known
        ]
        prepared = []
        for offset in range(0, len(candidates), 24):
            batch = candidates[offset : offset + 24]
            matrix = self._encode([row["body"] for row in batch])
            prepared.extend(
                (row["id"], len(vector), vector.tobytes())
                for row, vector in zip(batch, matrix)
            )
        # Replacement is one transaction, after every vector is validated.
        return self.repository.put_vectors(identity, prepared, replace=rebuild)

    def search(self, question, count=5, floor=0.2, excluded=()):
        if not question.strip() or count <= 0:
            return []
        query = self._encode([question])[0]
        sources, vectors = [], []
        for row in self.repository.vectors(self.encoder.identity):
            if row["source"] in excluded:
                continue
            try:
                vector = np.frombuffer(row["payload"], dtype=np.float32)
            except ValueError as error:
                raise ComponentError(
                    "Semantic index contains a damaged payload; rebuild it"
                ) from error
            if (
                len(vector) != len(query)
                or row["dimensions"] != len(query)
                or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1, atol=0.001)
            ):
                raise ComponentError(
                    "Semantic index is incompatible or corrupt; rebuild it"
                )
            vectors.append(vector)
            sources.append(row)
        if not sources:
            return []
        scores = np.asarray(vectors) @ query
        ranked = sorted(
            range(len(scores)), key=lambda i: (-float(scores[i]), sources[i]["source"])
        )
        return [
            Evidence(
                sources[i]["source"],
                sources[i]["conversation_key"],
                sources[i]["speaker"],
                sources[i]["body"],
                float(scores[i]),
            )
            for i in ranked[:count]
            if scores[i] >= floor
        ]


def prompt_evidence(sources, budget):
    records, remaining = [], max(0, budget)
    for source in sources:
        metadata = {
            "source": source.source_id,
            "conversation": source.conversation,
            "speaker": source.speaker,
        }
        overhead = len(json.dumps({**metadata, "quote": ""}, ensure_ascii=True)) + 1
        available = min(700, remaining - overhead)
        if available < 1:
            continue
        # JSON escaping is included in the actual budget check.
        quote = source.text[:available]
        while quote:
            line = json.dumps({**metadata, "quote": quote}, ensure_ascii=True)
            if len(line) + 1 <= remaining:
                records.append(line)
                remaining -= len(line) + 1
                break
            quote = quote[: len(quote) // 2]
    return "\n".join(records)
