from .configuration import Settings
from .repository import Repository
from .encoding import SentenceEncoder
from .recognition import WhisperRecognizer
from .inference import OllamaGenerator
from .synthesis import speech_service
from .semantic import SemanticMemory
from .vocabulary import Vocabulary
from .dialogue import Dialogue


def build(settings: Settings):
    repository = Repository(settings.database)
    return Dialogue(
        repository,
        SemanticMemory(repository, SentenceEncoder(settings)),
        Vocabulary(repository),
        WhisperRecognizer(settings),
        OllamaGenerator(settings),
        speech_service(settings),
        settings,
    )
