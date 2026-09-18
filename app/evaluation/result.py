from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    question: str
    expected_answer: str
    actual_answer: str
    score: float