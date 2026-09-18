import re


class ContextRelevance:
    _stop_words = {
        "a",
        "an",
        "and",
        "are",
        "is",
        "of",
        "the",
        "to",
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
    }

    def evaluate(
        self,
        question: str,
        contexts: list[str],
    ) -> float:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not contexts:
            return 0.0

        question_terms = self._tokenize(question)

        if not question_terms:
            return 0.0

        relevant_count = 0

        for context in contexts:
            context_terms = set(
                self._tokenize(context)
            )

            if any(
                term in context_terms
                for term in question_terms
            ):
                relevant_count += 1

        return relevant_count / len(contexts)

    def _tokenize(self, text: str) -> list[str]:
        return [
            term
            for term in re.findall(
                r"\w+",
                text.lower(),
            )
            if term not in self._stop_words
        ]