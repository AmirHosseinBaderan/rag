from app.rag.multi_query_retriever import MultiQueryRetriever


class FakeQueryGenerator:
    def generate(self, query: str) -> list[str]:
        return [
            "query one",
            "query two",
        ]


class FakeRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int,
        score_threshold=None,
        metadata_filter=None,
    ) -> list[dict]:
        results = {
            "query one": [
                {
                    "id": "chunk-1",
                    "score": 0.9,
                    "metadata": {"text": "one"},
                },
                {
                    "id": "chunk-2",
                    "score": 0.8,
                    "metadata": {"text": "two"},
                },
            ],
            "query two": [
                {
                    "id": "chunk-2",
                    "score": 0.95,
                    "metadata": {"text": "two"},
                },
                {
                    "id": "chunk-3",
                    "score": 0.7,
                    "metadata": {"text": "three"},
                },
            ],
        }

        return results[query]


def test_multi_query_retriever_merges_and_deduplicates_results():
    retriever = FakeRetriever()
    generator = FakeQueryGenerator()

    multi_retriever = MultiQueryRetriever(
        retriever=retriever,
        query_generator=generator,
    )

    results = multi_retriever.retrieve(
        query="original query",
        top_k=3,
    )

    assert len(results) == 3
    assert [result["id"] for result in results] == [
        "chunk-2",
        "chunk-1",
        "chunk-3",
    ]


def test_multi_query_retriever_rejects_empty_query():
    multi_retriever = MultiQueryRetriever(
        retriever=FakeRetriever(),
        query_generator=FakeQueryGenerator(),
    )

    try:
        multi_retriever.retrieve(
            query=" ",
            top_k=3,
        )
        assert False
    except ValueError as error:
        assert str(error) == "Query cannot be empty"


def test_multi_query_retriever_rejects_invalid_top_k():
    multi_retriever = MultiQueryRetriever(
        retriever=FakeRetriever(),
        query_generator=FakeQueryGenerator(),
    )

    try:
        multi_retriever.retrieve(
            query="test",
            top_k=0,
        )
        assert False
    except ValueError as error:
        assert str(error) == "top_k must be greater than zero"