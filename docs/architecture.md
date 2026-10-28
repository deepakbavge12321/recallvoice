# Architecture: a transaction boundary around a dialogue turn

## Decisions

1. Use a Python package with small constructor-injected services. The domain orchestrator depends on typed protocols, not Streamlit or a model framework. The UI is an adapter.
2. SQLite is the system of record for conversations, attempts, utterances, vectors, aliases, pronunciation preferences and observations. Each successful attempt commits its two utterances in one transaction. A unique client submission token provides replay protection. Failed attempts carry errors without contaminating dialogue history.
3. Semantic indexing is derived state. Model identity includes an optional revision and local weights fingerprint. A complete candidate rebuild is computed before a single transactional replacement, avoiding destructive partial rebuilds. SQL foreign keys enforce source lifetimes. For a personal corpus, batched NumPy cosine ranking is easier to audit than a vector server.
4. Recognition rules and pronunciation preferences are separate tables and services, so spelling correction never implies a speech preference. A compiled literal alternation performs one Unicode-boundary-aware pass. Chains are rejected at write time.
5. Native offline speech runs in a bounded child process to isolate OS speech event loops. Optional Edge TTS runs in a bounded worker; failure returns a typed provider failure and activates the next provider. Every audio result declares its MIME type.
6. A turn reads recent history and retrieves memories before generation. Only after a valid answer is received are both utterances committed. Embedding or speech failure after that point degrades the completed text turn rather than rolling it back.
7. Configuration is loaded from TOML with explicit environment overrides. Paths are resolved against the selected config file, independent of shell working directory. Runtime artifacts are excluded from version control.
8. Evaluation has its own newly authored corpus. It scores expected source IDs, records cold indexing separately from warmed retrieval, and never assigns scores to unavailable models. Real voice tests and fake-adapter tests have separate result labels.

## Interfaces

- Recognizer.transcribe(wav: bytes) -> str
- Generator.reply(messages: Sequence[ChatMessage]) -> str
- Encoder.encode(texts: Sequence[str]) -> float32 matrix; identity: str
- Synthesizer.render(text: str) -> AudioClip
- Repository: create/list/delete conversations; attempt lifecycle; atomic dialogue commit; source/vector/rule/observation access.
- SemanticMemory: synchronize(), rebuild(), search(query, excluded_ids), prompt_evidence(hits).
- Dialogue.submit(conversation_id, submission_id, text|wav, speak, pronunciation) -> TurnOutcome.

## Schema version 1

conversation → attempt → utterance → vector; attempt → observation.
recognition_rule and spoken_term are durable preferences independent of conversations.
Deletion cascades for derived records. Deleting a user utterance also redacts its raw/corrected duplicate in the attempt; its submission digest remains for replay protection. Vector rows are unique per source and encoder identity. Source IDs are integer keys; conversations and attempts use UUID text keys.

## Flow

Input → recognition (audio only) → rule correction → retrieval + recent history → local generation → atomic dialogue commit → vector synchronization → optional spoken respelling + synthesis.

A stage recorder writes outcome and milliseconds even on exceptions. Attempt status distinguishes running, failed and completed; an interrupted running attempt requires a new submission token. The UI sends explicit submission tokens and does not automatically resubmit recordings on reruns.

## Public implementation references

- https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input
- https://github.com/SYSTRAN/faster-whisper
- https://sbert.net/docs/package_reference/sentence_transformer/model.html
- https://docs.ollama.com/api/chat
- https://pyttsx3.readthedocs.io/en/latest/engine.html
- https://docs.python.org/3/library/sqlite3.html
