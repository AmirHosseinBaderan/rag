from app.generation.llm import LLM


class FakeLLM(LLM):
    def generate(self, prompt: str) -> str:
        return "This is a generated answer."


def test_fake_llm_generates_answer() -> None:
    llm = FakeLLM()

    answer = llm.generate(
        "What is FastAPI?"
    )

    assert answer == "This is a generated answer."