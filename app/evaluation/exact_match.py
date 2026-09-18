class ExactMatch:
    def evaluate(
        self,
        expected: str,
        actual: str,
    ) -> float:
        return float(expected.strip() == actual.strip())