from app.evaluation.dataset import EvaluationCase, EvaluationDataset


def test_evaluation_dataset_contains_cases() -> None:
    dataset = EvaluationDataset(
        cases=[
            EvaluationCase(
                question="What is FastAPI?",
                expected_answer="FastAPI is a Python web framework.",
            ),
        ]
    )

    assert len(dataset.cases) == 1
    assert dataset.cases[0].question == "What is FastAPI?"
    assert (
        dataset.cases[0].expected_answer
        == "FastAPI is a Python web framework."
    )


def test_evaluation_case_rejects_empty_question() -> None:
    try:
        EvaluationCase(
            question="",
            expected_answer="Some answer",
        )
        assert False
    except ValueError:
        pass


def test_evaluation_case_rejects_empty_expected_answer() -> None:
    try:
        EvaluationCase(
            question="What is FastAPI?",
            expected_answer="",
        )
        assert False
    except ValueError:
        pass