from app.evaluation.dataset import EvaluationDataset
from app.evaluation.exact_match import ExactMatch
from app.evaluation.report import EvaluationReport
from app.evaluation.result import EvaluationResult


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
    ) -> EvaluationReport:
        results: list[EvaluationResult] = []

        for case in dataset.cases:
            actual_answer = self._rag.ask(
                case.question
            )

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

        return EvaluationReport(results)