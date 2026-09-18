import re

from app.evaluation.metrics.cosine_similarity import CosineSimilarity
from app.retrieval.embedder import Embedder


class SemanticFaithfulness:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._similarity = CosineSimilarity()

    def calculate(
        self,
        answer: str,
        contexts: list[str],
    ) -> float:
        if not answer.strip():
            raise ValueError("Answer cannot be empty")

        if not contexts:
            raise ValueError("Contexts cannot be empty")

        if any(not context.strip() for context in contexts):
            raise ValueError("Context cannot be empty")

        answer_sentences = self._split_sentences(answer)

        context_vectors = [
            self._embedder.embed(context)
            for context in contexts
        ]

        scores: list[float] = []

        for sentence in answer_sentences:
            sentence_vector = self._embedder.embed(sentence)

            best_score = max(
                self._similarity.calculate(
                    sentence_vector,
                    context_vector,
                )
                for context_vector in context_vectors
            )

            scores.append(best_score)

        return sum(scores) / len(scores)

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]