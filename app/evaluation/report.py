from app.evaluation.result import EvaluationResult


class EvaluationReport:
    def __init__(
        self,
        results: list[EvaluationResult],
    ) -> None:
        self._results = results

    @property
    def results(self) -> list[EvaluationResult]:
        return self._results

    @property
    def average_score(self) -> float:
        if not self._results:
            return 0.0

        return sum(
            result.score
            for result in self._results
        ) / len(self._results)