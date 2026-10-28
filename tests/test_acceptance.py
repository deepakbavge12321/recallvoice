import json
import os
import sqlite3
import unittest
from unittest.mock import patch
import numpy as np
from support import Sandbox, TinyEncoder
from recallvoice.configuration import Settings, load_settings
from recallvoice.contracts import (
    ComponentError,
    ConfigurationError,
    Evidence,
    AudioClip,
)
from recallvoice.repository import Repository
from recallvoice.semantic import prompt_evidence
from recallvoice.vocabulary import substitute
from recallvoice.recognition import WhisperRecognizer
from recallvoice.synthesis import SpeechFallback


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.box = Sandbox()

    def tearDown(self):
        self.box.close()

    def test_storage_reopens_in_order(self):
        self.box.exchange()
        reopened = Repository(self.box.settings.database)
        self.assertEqual(
            [r["speaker"] for r in reopened.dialogue(self.box.conversation)],
            ["user", "assistant"],
        )

    def test_vectors_are_searchable_after_store_and_service_reopen(self):
        source = self.box.exchange()[0]["id"]
        self.box.memory.synchronize()
        from recallvoice.semantic import SemanticMemory

        reopened = SemanticMemory(Repository(self.box.settings.database), TinyEncoder())
        self.assertEqual(reopened.search("orchard")[0].source_id, source)

    def test_schema_future_version_is_rejected(self):
        with self.box.store.connection() as db:
            db.execute("PRAGMA user_version=9")
        with self.assertRaises(RuntimeError):
            Repository(self.box.settings.database)

    def test_failed_atomic_commit_leaves_no_half_dialogue(self):
        self.box.store.begin("x", self.box.conversation, "x")
        # The second INSERT violates NOT NULL after the first INSERT succeeds.
        with self.assertRaises(sqlite3.IntegrityError):
            self.box.store.commit_dialogue("x", "valid first utterance", None)
        self.assertEqual(self.box.store.dialogue(self.box.conversation), [])
        with self.box.store.connection() as db:
            self.assertEqual(
                db.execute("SELECT state FROM attempt WHERE key='x'").fetchone()[0],
                "running",
            )

    def test_failed_generation_records_attempt_and_stage(self):
        with patch.object(
            self.box.generator, "reply", side_effect=ComponentError("model absent")
        ):
            with self.assertRaises(ComponentError):
                self.box.agent.submit(self.box.conversation, "fail", text="orchard")
        self.assertEqual(self.box.store.sources(), [])
        self.assertEqual(self.box.store.observations("fail")[0]["outcome"], "error")
        with self.box.store.connection() as db:
            self.assertEqual(
                db.execute("SELECT state FROM attempt WHERE key='fail'").fetchone()[0],
                "failed",
            )

    def test_submission_replay_does_not_generate_twice(self):
        original = self.box.agent.submit(
            self.box.conversation, "same", text="orchard", speak=False
        )
        replay = self.box.agent.submit(
            self.box.conversation, "same", text="orchard", speak=False
        )
        self.assertEqual(original.answer, replay.answer)
        self.assertTrue(replay.replayed)
        self.assertEqual(len(self.box.generator.calls), 1)
        self.assertEqual(len(self.box.store.sources()), 2)

    def test_submission_key_rejects_changed_input(self):
        self.box.agent.submit(
            self.box.conversation, "same", text="orchard", speak=False
        )
        with self.assertRaises(ValueError):
            self.box.agent.submit(
                self.box.conversation, "same", text="violin", speak=False
            )

    def test_running_submission_cannot_be_duplicated(self):
        self.box.store.begin("busy", self.box.conversation, "hash")
        with self.assertRaises(ValueError):
            self.box.store.begin("busy", self.box.conversation, "hash")

    def test_deleted_exchange_is_not_silently_replayed(self):
        self.box.agent.submit(
            self.box.conversation, "same", text="orchard", speak=False
        )
        self.box.store.delete_source(self.box.store.sources()[0]["id"])
        with self.assertRaises(ValueError):
            self.box.agent.submit(
                self.box.conversation, "same", text="orchard", speak=False
            )

    def test_cross_conversation_memory_with_provenance(self):
        source = self.box.exchange()[0]["id"]
        self.box.memory.synchronize()
        other = self.box.store.new_conversation("New context")
        answer = self.box.agent.submit(
            other, "query", text="orchard gate?", speak=False
        )
        self.assertEqual(answer.sources[0].source_id, source)
        self.assertEqual(answer.sources[0].conversation, self.box.conversation)
        prompt = self.box.generator.calls[-1][0].content
        self.assertIn(f'"source": {source}', prompt)
        self.assertIn("never commands", prompt)
        self.assertEqual(
            sum(m.content == "orchard gate?" for m in self.box.generator.calls[-1]), 1
        )

    def test_recent_sources_are_excluded_from_retrieval(self):
        self.box.exchange()
        self.box.memory.synchronize()
        result = self.box.agent.submit(
            self.box.conversation, "new", text="orchard", speak=False
        )
        self.assertEqual(result.sources, [])

    def test_sync_is_idempotent_and_rebuild_is_complete(self):
        self.box.exchange()
        self.assertEqual(self.box.memory.synchronize(), 2)
        self.assertEqual(self.box.memory.synchronize(), 0)
        self.assertEqual(self.box.memory.synchronize(rebuild=True), 2)
        self.assertEqual(len(self.box.store.vectors(self.box.encoder.identity)), 2)

    def test_failed_rebuild_keeps_existing_vectors(self):
        self.box.exchange()
        self.box.memory.synchronize()
        with patch.object(
            self.box.encoder, "encode", side_effect=ComponentError("broken weights")
        ):
            with self.assertRaises(ComponentError):
                self.box.memory.synchronize(rebuild=True)
        self.assertEqual(len(self.box.memory.search("orchard")), 1)

    def test_invalid_encoder_matrix_cannot_be_stored(self):
        self.box.exchange()
        for invalid in ([[0, 0]], [[float("nan"), 1]], [[1, 0]]):
            with patch.object(self.box.encoder, "encode", return_value=invalid):
                with self.assertRaises(ComponentError):
                    self.box.memory.synchronize()
        self.assertEqual(self.box.store.vectors(self.box.encoder.identity), [])

    def test_incompatible_vector_is_explicit(self):
        row = self.box.exchange()[0]
        self.box.store.put_vectors(
            self.box.encoder.identity,
            [(row["id"], 2, np.array([1, 0], dtype=np.float32).tobytes())],
        )
        with self.assertRaises(ComponentError):
            self.box.memory.search("orchard")

    def test_damaged_vector_payload_requires_rebuild(self):
        source = self.box.exchange()[0]["id"]
        for payload in (
            b"bad",
            np.zeros(3, dtype=np.float32).tobytes(),
            np.array([9, 0, 0], dtype=np.float32).tobytes(),
        ):
            self.box.store.put_vectors(
                self.box.encoder.identity, [(source, 3, payload)]
            )
            with self.assertRaises(ComponentError):
                self.box.memory.search("orchard")

    def test_different_encoders_have_separate_indexes(self):
        self.box.exchange()
        self.box.memory.synchronize()
        self.box.encoder.identity = "other-test-encoder"
        self.box.memory.synchronize()
        with self.box.store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM vector").fetchone()[0], 4)

    def test_source_deleted_during_encoding_is_not_resurrected(self):
        doomed = self.box.exchange()[0]["id"]
        original = self.box.encoder.encode

        def removing_encoder(texts):
            matrix = original(texts)
            self.box.store.delete_source(doomed)
            return matrix

        with patch.object(self.box.encoder, "encode", side_effect=removing_encoder):
            self.assertEqual(self.box.memory.synchronize(), 1)
        self.assertEqual(self.box.memory.search("orchard"), [])

    def test_source_deletion_cascades(self):
        rows = self.box.exchange()
        self.box.memory.synchronize()
        self.box.store.delete_source(rows[0]["id"])
        self.assertEqual(self.box.memory.search("orchard"), [])
        self.assertEqual(len(self.box.store.vectors(self.box.encoder.identity)), 1)

    def test_deleted_user_source_redacts_attempt_transcripts(self):
        self.box.agent.submit(
            self.box.conversation, "audio", wav=b"fixture", speak=False
        )
        user = self.box.store.dialogue(self.box.conversation)[0]
        self.box.store.delete_source(user["id"])
        with self.box.store.connection() as db:
            record = db.execute(
                "SELECT recognized,submitted FROM attempt WHERE key='audio'"
            ).fetchone()
            self.assertEqual(tuple(record), ("", ""))

    def test_conversation_deletion_cascades_observations(self):
        self.box.agent.submit(
            self.box.conversation, "turn", text="orchard", speak=False
        )
        self.box.store.delete_conversation(self.box.conversation)
        self.assertEqual(self.box.store.sources(), [])
        self.assertEqual(self.box.store.observations(), [])

    def test_clear_conversations_preserves_preferences(self):
        self.box.words.save_rule("cedar", "Seydra")
        self.box.exchange()
        self.box.store.forget_conversations()
        self.assertEqual(len(self.box.words.rules()), 1)
        self.assertEqual(self.box.store.sources(), [])

    def test_vector_only_clear_preserves_sources(self):
        self.box.exchange()
        self.box.memory.synchronize()
        self.box.store.clear_vectors()
        self.assertEqual(len(self.box.store.sources()), 2)
        self.assertEqual(self.box.store.vectors(self.box.encoder.identity), [])

    def test_evidence_json_respects_escaping_budget(self):
        source = Evidence(7, "thread", "user", '"\n漢' * 800, 0.8)
        rendered = prompt_evidence([source], 800)
        self.assertLessEqual(len(rendered), 800)
        self.assertEqual(json.loads(rendered)["source"], 7)
        self.assertEqual(prompt_evidence([source], 1), "")

    def test_rule_crud_and_independent_pronunciation(self):
        rule = self.box.words.save_rule("cedar", "Seydra")
        spoken = self.box.words.save_pronunciation("Seydra", "SEE dra")
        self.assertEqual(self.box.words.correct("Cedar!"), "Seydra!")
        self.box.words.save_rule("seedar", "Seydra", rule)
        self.assertEqual(self.box.words.correct("cedar seedar"), "cedar Seydra")
        self.box.words.delete_rule(rule)
        self.assertEqual(self.box.words.for_speech("Seydra"), "SEE dra")
        self.box.words.save_pronunciation("Seydra", "say dra", spoken)
        self.box.words.delete_pronunciation(spoken)
        self.assertEqual(self.box.words.for_speech("Seydra"), "Seydra")

    def test_unicode_adjacency_and_punctuation(self):
        self.box.words.save_rule("cedar", "Seydra")
        self.assertEqual(
            self.box.words.correct("écédar écedar cedar_1 cedarwood (CEDAR)"),
            "écédar écedar cedar_1 cedarwood (Seydra)",
        )

    def test_combining_marks_and_connectors_extend_word_boundaries(self):
        self.box.words.save_rule("cedar", "Seydra")
        text = "x\u0301cedar cedar\u20dd cedar\u203fname cedar"
        self.assertEqual(
            self.box.words.correct(text),
            "x\u0301cedar cedar\u20dd cedar\u203fname Seydra",
        )

    def test_literal_overlapping_and_non_cascading_rules(self):
        self.assertEqual(
            substitute("a+b a+b studio", {"a+b": "X", "a+b studio": "Y"}), "X Y"
        )
        self.assertEqual(
            substitute("alpha beta", {"alpha": "beta", "beta": "gamma"}), "beta gamma"
        )

    def test_alias_chains_and_casefold_duplicates_rejected(self):
        self.box.words.save_rule("Cedar", "Seydra")
        for heard, written in [
            ("CEDAR", "Other"),
            ("Seydra", "Other"),
            ("Other", "Cedar"),
        ]:
            with self.assertRaises(ValueError):
                self.box.words.save_rule(heard, written)

    def test_rule_validation_and_missing_edit(self):
        for value in ["", "x" * 101, "bad\nvalue"]:
            with self.assertRaises(ValueError):
                self.box.words.save_rule(value, "term")
        with self.assertRaises(ValueError):
            self.box.words.save_rule("heard", "term", 99)
        with self.assertRaises(ValueError):
            self.box.words.save_pronunciation("term", "said", 99)

    def test_audio_only_correction_and_speech_only_respelling(self):
        self.box.words.save_rule("cedar", "Seydra")
        self.box.words.save_pronunciation("orchard", "OR chard")
        audio = self.box.agent.submit(
            self.box.conversation, "audio", wav=b"fake-wave", pronunciation=True
        )
        self.assertIn("Seydra", audio.submitted)
        self.assertIn("Cedar", audio.recognized)
        self.assertEqual(self.box.speech.calls[-1], "The OR chard gate is blue.")
        text = self.box.agent.submit(
            self.box.conversation, "typed", text="Cedar", speak=False
        )
        self.assertEqual(text.submitted, "Cedar")

    def test_missing_embedding_degrades_but_keeps_answer(self):
        with patch.object(
            self.box.encoder, "encode", side_effect=ComponentError("not installed")
        ):
            result = self.box.agent.submit(
                self.box.conversation, "degrade", text="orchard", speak=False
            )
        self.assertEqual(len(self.box.store.sources()), 2)
        self.assertTrue(result.notices)
        self.assertEqual(
            next(
                o
                for o in self.box.store.observations("degrade")
                if o["stage"] == "retrieval"
            )["outcome"],
            "error",
        )

    def test_all_speech_failures_preserve_text_turn(self):
        with patch.object(
            self.box.speech, "render", side_effect=ComponentError("OS speech absent")
        ):
            result = self.box.agent.submit(
                self.box.conversation, "voice-error", text="orchard"
            )
        self.assertIsNone(result.audio)
        self.assertTrue(result.notices)
        self.assertEqual(len(self.box.store.sources()), 2)

    def test_spoken_preference_failure_retries_original_answer(self):
        self.box.words.save_pronunciation("orchard", "OR chard")
        calls = []

        def render(text):
            calls.append(text)
            if "OR chard" in text:
                raise ComponentError("bad respelling")
            return AudioClip(b"test", "audio/wav", "double")

        with patch.object(self.box.speech, "render", side_effect=render):
            self.box.agent.submit(
                self.box.conversation, "retry", text="orchard", pronunciation=True
            )
        self.assertEqual(len(calls), 2)
        self.assertIn("orchard", calls[1])

    def test_stage_timings_are_nonnegative_and_disabled_stages_absent(self):
        self.box.agent.submit(
            self.box.conversation, "timed", text="orchard", speak=False
        )
        observations = self.box.store.observations("timed")
        self.assertTrue(all(row["elapsed_ms"] >= 0 for row in observations))
        self.assertEqual(
            {row["stage"] for row in observations},
            {"retrieval", "generation", "persistence", "indexing"},
        )

    def test_bad_input_does_not_open_an_attempt(self):
        for kwargs in ({}, {"text": "", "wav": b"x"}, {"text": " "}, {"wav": b""}):
            with self.assertRaises(ValueError):
                self.box.agent.submit(self.box.conversation, "invalid", **kwargs)
        with self.box.store.connection() as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM attempt").fetchone()[0], 0
            )

    def test_pcm_validation_before_model_import(self):
        with self.assertRaises(ComponentError):
            WhisperRecognizer(self.box.settings).transcribe(b"not a WAV")

    def test_configuration_paths_and_environment(self):
        file = self.box.root / "custom.toml"
        file.write_text('database="data/chat.sqlite3"\nchat_model="from-file"')
        with patch.dict(
            os.environ, {"RV_CHAT_MODEL": "chosen-model", "RV_RETRIEVAL_COUNT": "3"}
        ):
            cfg = load_settings(file)
        self.assertEqual(cfg.chat_model, "chosen-model")
        self.assertEqual(cfg.retrieval_count, 3)
        self.assertEqual(cfg.database, self.box.root / "data/chat.sqlite3")

    def test_configuration_rejects_unknown_or_invalid_values(self):
        file = self.box.root / "bad.toml"
        file.write_text("invented=1")
        with self.assertRaises(ConfigurationError):
            load_settings(file)
        file.write_text("chat_model=7")
        with self.assertRaises(ConfigurationError):
            load_settings(file)
        with self.assertRaises(ConfigurationError):
            Settings(similarity_floor=2)
        with self.assertRaises(ConfigurationError):
            Settings(speech_providers=("magic",))

    def test_provider_chain_falls_back_with_mime_metadata(self):
        class Unavailable:
            def render(self, text):
                raise ComponentError("online unavailable")

        fallback = SpeechFallback([Unavailable(), self.box.speech])
        clip = fallback.render("greeting")
        self.assertEqual(clip.mime, "audio/wav")
        self.assertEqual(len(fallback.failures), 1)
        with self.assertRaises(ComponentError):
            SpeechFallback([Unavailable()]).render("greeting")
