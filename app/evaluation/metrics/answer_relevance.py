from app.evaluation.metrics.cosine_similarity import CosineSimilarity
from app.retrieval.embedder import Embedder


class AnswerRelevance:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._similarity = CosineSimilarity()

    def calculate(
        self,
        question: str,
        answer: str,
    ) -> float:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not answer.strip():
            raise ValueError("Answer cannot be empty")

        question_vector = self._embedder.embed(question)
        answer_vector = self._embedder.embed(answer)

        return self._similarity.calculate(
            question_vector,
            answer_vector,
        )