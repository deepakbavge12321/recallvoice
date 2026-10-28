# RecallVoice requirements and acceptance contract

Behavior and acceptance criteria for the voice assistant.

## Behavior

R1. A user can submit typed text, browser microphone recordings or WAV files. Whisper produces the raw recognized transcript; recognition corrections are applied only to audio input.
R2. Ollama generates an answer using a configured local model (initial target qwen2.5:1.5b). Network, malformed-response and model-not-installed failures are explicit.
R3. Speech generation returns encoded audio plus provider/MIME metadata. Offline OS speech is available; failure of an optional online provider falls through to offline speech. Text remains usable when all speech providers fail.
R4. A selectable, durable conversation has ordered user/assistant utterances. An answer failure must not leave a half-completed conversation exchange. Failed attempts and timing records remain observable.
R5. User and assistant utterances have independently addressable semantic vectors. Retrieval spans conversations and returns ranked provenance-bearing sources within a prompt budget. Reindexing is idempotent and atomic: a failed rebuild preserves the old index.
R6. Recognition aliases map one literal phrase to a canonical term; separate pronunciation preferences respell terms for speech only. CRUD validates collisions, ambiguous chains and length limits. Matching is Unicode-aware, longest-first, single-pass, and cannot rewrite substrings inside unrelated words.
R7. Users can delete individual utterances, whole conversations, all conversation memory, or just vector indexes. Derived vectors cannot outlive deleted source records.
R8. The Streamlit UI offers chat/audio input, conversation switching, source inspection, alias/pronunciation editing, memory administration and evaluation.
R9. Persist duration, outcome and attempt ID for recognition, retrieval, generation, persistence, indexing and speech, including failures. Missing/disabled stages are distinct from zero time.
R10. Fixed-data benchmarks and real two-conversation voice tests write fresh machine-readable evidence, recording model/configuration and dataset identity. Test doubles are never presented as real model measurements.

## Acceptance cases

A1. Restart a store, reopen a conversation, recover chronological utterances.
A2. Simulated generator failure stores an attempt/error but no partial dialogue.
A3. Repeated submission ID returns the existing completed exchange; no duplicate utterances or generation call.
A4. A paraphrased question in a different conversation retrieves the expected source; the prompt includes source IDs and treats memories as quoted evidence.
A5. Reindex twice without duplicates; failed rebuild leaves prior vectors searchable; deleted sources disappear from all model indexes.
A6. Alias correction handles punctuation, Unicode adjacency, literal regex symbols, overlapping phrases and case, without cascading changes. Typed input remains exact.
A7. Recognition aliases and speech-only preferences can be independently edited/deleted.
A8. Missing embedding weights degrades to recent-history chat with an explicit notice. Missing Whisper or Ollama raises a visible, recorded error. TTS failure preserves the textual answer.
A9. All interface paths run in isolated unit/integration tests; browser microphone acquisition requires a separate hardware check.
A10. A real synthetic WAV states a previously unused project fact; a second conversation asks about it. Both transcribe, aliases change observed variants, retrieval attributes the first conversation, the local LLM recalls the fact, and both synthesized outputs decode into non-silent audio.

## Non-goals

No hosted LLM, autonomous tools, biometric speaker identity, phoneme-learning network, multi-user authorization, or guaranteed pronunciation.

