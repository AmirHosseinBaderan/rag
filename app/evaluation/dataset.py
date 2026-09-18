from dataclasses import dataclass

@dataclass(frozen=True)
class EvaluationCase:
    question:str
    expected_answer: str

    def __post_init__(self):
        if not self.question.strip():
            raise ValueError("Question cannot be empty")

        if not self.expected_answer.strip():
            raise ValueError("Expected answer cannot be empty ")


@dataclass(frozen=True)
class EvaluationDataset:
    cases:list[EvaluationCase]