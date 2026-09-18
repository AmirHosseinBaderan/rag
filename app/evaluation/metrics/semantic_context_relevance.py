from app.evaluation.metrics.cosine_similarity import CosineSimilarity
from app.retrieval.embedder import Embedder


class SemanticContextRelevance:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._similarity = CosineSimilarity()

    def evaluate(
        self,
        question: str,
        contexts: list[str],
    ) -> float:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not contexts:
            raise ValueError("Contexts cannot be empty")

        question_vector = self._embedder.embed(question)

        scores = [
            self._similarity.calculate(
                question_vector,
                self._embedder.embed(context),
            )
            for context in contexts
        ]

        return sum(scores) / len(scores)