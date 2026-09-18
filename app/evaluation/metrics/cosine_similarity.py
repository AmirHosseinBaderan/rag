import math


class CosineSimilarity:
    def calculate(
        self,
        first: list[float],
        second: list[float],
    ) -> float:
        if len(first) != len(second):
            raise ValueError("Vectors must have the same dimension")

        if not first:
            raise ValueError("Vectors cannot be empty")

        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))

        if first_norm == 0 or second_norm == 0:
            raise ValueError("Zero vector is not allowed")

        dot_product = sum(
            first_value * second_value
            for first_value, second_value in zip(first, second)
        )

        return dot_product / (first_norm * second_norm)