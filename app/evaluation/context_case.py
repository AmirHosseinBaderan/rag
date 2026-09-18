from dataclasses import dataclass


@dataclass(frozen=True)
class ContextEvaluationCase:
    question: str
    contexts: list[str]

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("Question cannot be empty")

        if not self.contexts:
            raise ValueError("Contexts cannot be empty")

        if any(not context.strip() for context in self.contexts):
            raise ValueError("Context cannot be empty")