from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @abstractmethod
    def clean_text(self, text: str) -> str:
        pass

    @abstractmethod
    def answer_question(self, question: str, context: str) -> str:
        pass
