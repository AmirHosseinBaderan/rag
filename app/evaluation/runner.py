from dataclasses import dataclass

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.exact_match import ExactMatch


@dataclass(frozen=True)
class EvaluationResult:
    question: str
    expected_answer: str
    actual_answer: str
    score: float


class EvaluationRunner:
    def __init__(
        self,
        rag,
        evaluator: ExactMatch,
    ) -> None:
        self._rag = rag
        self._evaluator = evaluator

    def run(
        self,
        dataset: EvaluationDataset,
    ) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []

        for case in dataset.cases:
            actual_answer = self._rag.ask(case.question)

            score = self._evaluator.evaluate(
                expected=case.expected_answer,
                actual=actual_answer,
            )

            results.append(
                EvaluationResult(
                    question=case.question,
                    expected_answer=case.expected_answer,
                    actual_answer=actual_answer,
                    score=score,
                )
            )

        return results