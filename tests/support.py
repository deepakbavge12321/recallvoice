from pathlib import Path
import tempfile
import numpy as np
from recallvoice.configuration import Settings
from recallvoice.contracts import AudioClip
from recallvoice.repository import Repository
from recallvoice.semantic import SemanticMemory
from recallvoice.vocabulary import Vocabulary
from recallvoice.dialogue import Dialogue


class TinyEncoder:
    identity = "acceptance-double-encoder"

    def encode(self, texts):
        return np.array(
            [
                [1, 0, 0]
                if "orchard" in text.lower()
                else [0, 1, 0]
                if "violin" in text.lower()
                else [0, 0, 1]
                for text in texts
            ],
            dtype=np.float32,
        )


class ReplyDouble:
    def __init__(self):
        self.calls = []

    def reply(self, messages):
        self.calls.append(messages)
        return "The orchard gate is blue."


class RecognitionDouble:
    def transcribe(self, wav):
        return "Please open Cedar orchard."


class SpeechDouble:
    def __init__(self):
        self.calls = []

    def render(self, text):
        self.calls.append(text)
        return AudioClip(b"test-only-audio", "audio/wav", "double")


class Sandbox:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings = Settings(
            database=self.root / "test.sqlite3", model_dir=self.root / "models"
        )
        self.store = Repository(self.settings.database)
        self.encoder = TinyEncoder()
        self.generator = ReplyDouble()
        self.speech = SpeechDouble()
        self.memory = SemanticMemory(self.store, self.encoder)
        self.words = Vocabulary(self.store)
        self.agent = Dialogue(
            self.store,
            self.memory,
            self.words,
            RecognitionDouble(),
            self.generator,
            self.speech,
            self.settings,
        )
        self.conversation = self.store.new_conversation("Acceptance conversation")

    def close(self):
        self.temporary.cleanup()

    def exchange(self, text="The orchard gate is blue.", conversation=None, key="seed"):
        owner = conversation or self.conversation
        self.store.begin(key, owner, key)
        self.store.commit_dialogue(key, text, "Acknowledged.")
        return self.store.dialogue(owner)
