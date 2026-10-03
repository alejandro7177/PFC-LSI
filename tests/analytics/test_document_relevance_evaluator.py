import json
import pandas as pd
import pytest
from src.analytics.document_relevance_evaluator import DocumentRelevanceEvaluator
from src.strategies.base import BaseLLMStrategy


class FixedScoreLLMStrategy(BaseLLMStrategy):
    """Estrategia LLM de prueba con puntaje fijo."""

    def __init__(self, score: float = 0.8, reasoning: str = "Test reasoning"):
        self.score = score
        self.reasoning = reasoning

    @property
    def model_name(self) -> str:
        return "FixedScoreLLMTest"

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        return f'{{"sufficiency_score": {self.score}, "reasoning": "{self.reasoning}"}}'


@pytest.fixture
def sample_retrieval_df() -> pd.DataFrame:
    """DataFrame de prueba con preguntas, resúmenes e IDs recuperados."""
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "question": [
                "What methods analyze climate chaos?",
                "How to assimilate data in Lorenz models?",
            ],
            "doc_id_1": ["100", "200"],
            "abstract_1": ["Abstract 100 on climate chaos.", "Abstract 200 on Lorenz."],
            "doc_id_2": ["101", "201"],
            "abstract_2": ["Abstract 101 on climate chaos.", "Abstract 201 on Lorenz."],
            "gemma": [
                json.dumps(["100", "101", "500"]),
                json.dumps(["999", "888"]),
            ],
            "qwen": [
                json.dumps(["100", "999"]),
                json.dumps(["200", "201"]),
            ],
        }
    )


def test_parse_retrieved_list():
    evaluator = DocumentRelevanceEvaluator()
    parsed_json = evaluator.parse_retrieved_list('["10", "20", "30"]')
    assert parsed_json == ["10", "20", "30"]

    parsed_list = evaluator.parse_retrieved_list([10, 20])
    assert parsed_list == ["10", "20"]

    assert evaluator.parse_retrieved_list(None) == []


def test_compute_mrr():
    evaluator = DocumentRelevanceEvaluator()
    mrr = evaluator.compute_mrr(expected_ids=["100", "101"], retrieved_ids=["200", "101", "100"])
    assert mrr == 0.5


def test_parse_llm_response():
    evaluator = DocumentRelevanceEvaluator()
    res = evaluator.parse_llm_response('{"sufficiency_score": 0.85, "reasoning": "Good context"}')
    assert res["sufficiency_score"] == 0.85
    assert res["reasoning"] == "Good context"


def test_evaluate_hybrid_sufficiency(sample_retrieval_df):
    llm_strat = FixedScoreLLMStrategy(score=0.8, reasoning="Sufficient context")
    evaluator = DocumentRelevanceEvaluator(llm_strategy=llm_strat, doc_id_cols=["doc_id_1", "doc_id_2"])

    detailed_df, summary_df = evaluator.evaluate_hybrid_sufficiency(
        sample_retrieval_df,
        eval_mode="hybrid",
        top_k=20,
    )

    assert not detailed_df.empty
    assert not summary_df.empty

    # Gemma row 0: recall = 1.0 (100 and 101), llm_score = 0.8 => hybrid_score = 0.5*1.0 + 0.5*0.8 = 0.9
    assert detailed_df.loc[0, "gemma_recall"] == 1.0
    assert detailed_df.loc[0, "gemma_llm_score"] == 0.8
    assert detailed_df.loc[0, "gemma_score"] == 0.9

    # Qwen row 0: recall = 0.5 (100), llm_score = 0.8 => hybrid_score = 0.5*0.5 + 0.5*0.8 = 0.65
    assert detailed_df.loc[0, "qwen_recall"] == 0.5
    assert detailed_df.loc[0, "qwen_llm_score"] == 0.8
    assert detailed_df.loc[0, "qwen_score"] == 0.65

    assert detailed_df.loc[0, "retrieval_winner"] == "Gemma"
