import pytest

from app.evaluation.metrics.cosine_similarity import CosineSimilarity


def test_identical_vectors_have_similarity_one():
    similarity = CosineSimilarity()

    result = similarity.calculate([1.0, 0.0], [1.0, 0.0])

    assert result == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    similarity = CosineSimilarity()

    result = similarity.calculate([1.0, 0.0], [0.0, 1.0])

    assert result == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_minus_one():
    similarity = CosineSimilarity()

    result = similarity.calculate([1.0, 0.0], [-1.0, 0.0])

    assert result == pytest.approx(-1.0)


def test_vectors_must_have_same_dimension():
    similarity = CosineSimilarity()

    with pytest.raises(ValueError, match="same dimension"):
        similarity.calculate([1.0, 0.0], [1.0])


def test_zero_vector_is_not_allowed():
    similarity = CosineSimilarity()

    with pytest.raises(ValueError, match="[Zz]ero vector"):
        similarity.calculate([0.0, 0.0], [1.0, 0.0])