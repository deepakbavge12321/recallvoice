from pathlib import Path
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from support import Sandbox


class InterfaceIntegration(unittest.TestCase):
    def test_text_turn_conversation_switch_and_vocabulary_forms(self):
        box = Sandbox()
        try:
            with (
                patch("recallvoice.interface.services", return_value=box.agent),
                patch("recallvoice.interface.load_settings", return_value=box.settings),
            ):
                ui = AppTest.from_file(
                    str(Path(__file__).resolve().parents[1] / "ui.py"),
                    default_timeout=20,
                ).run()
                self.assertEqual(len(ui.exception), 0)
                self.assertEqual(
                    [t.label for t in ui.tabs],
                    ["Conversation", "Vocabulary", "Memory library", "Diagnostics"],
                )
                next(
                    w for w in ui.toggle if w.label == "Generate spoken replies"
                ).set_value(False).run()
                next(w for w in ui.text_area if w.label == "Message").set_value(
                    "orchard gate"
                )
                next(w for w in ui.button if w.label == "Submit turn").click().run()
                self.assertEqual(len(ui.exception), 0)
                self.assertEqual(len(box.generator.calls), 1)
                next(
                    w for w in ui.text_input if w.label == "Recognized phrase"
                ).set_value("cedar")
                next(
                    w for w in ui.text_input if w.label == "Canonical spelling"
                ).set_value("Seydra")
                next(
                    w for w in ui.button if w.label == "Save recognition rule"
                ).click().run()
                self.assertEqual(box.words.correct("cedar"), "Seydra")
                next(
                    w for w in ui.button if w.label == "Create conversation"
                ).click().run()
                self.assertEqual(len(ui.exception), 0)
                self.assertEqual(len(box.store.conversations()), 2)
        finally:
            box.close()

    def test_memory_library_search_delete_and_index_clear(self):
        box = Sandbox()
        box.exchange()
        try:
            with (
                patch("recallvoice.interface.services", return_value=box.agent),
                patch("recallvoice.interface.load_settings", return_value=box.settings),
            ):
                ui = AppTest.from_file(
                    str(Path(__file__).resolve().parents[1] / "ui.py"),
                    default_timeout=20,
                ).run()
                next(
                    w for w in ui.text_input if w.label == "Search all conversations"
                ).set_value("orchard")
                next(w for w in ui.button if w.label == "Search sources").click().run()
                self.assertEqual(len(ui.exception), 0)
                next(
                    w for w in ui.button if w.label == "Delete this source"
                ).click().run()
                self.assertEqual(len(box.store.sources()), 1)
                next(
                    w for w in ui.button if w.label == "Discard vector index only"
                ).click().run()
                self.assertEqual(box.store.vectors(box.encoder.identity), [])
                self.assertEqual(len(ui.exception), 0)
        finally:
            box.close()
