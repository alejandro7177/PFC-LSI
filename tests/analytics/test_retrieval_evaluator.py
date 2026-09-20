import json
from pathlib import Path
import pandas as pd
import pytest

from src.analytics.retrieval_evaluator import RetrievalEvaluator, RetrievalMetricsCalculator


class TestRetrievalMetricsCalculator:
    def test_reciprocal_rank(self):
        expected = ["10", "20", "30"]

        # Primer elemento relevante en pos 1 -> RR = 1.0
        retrieved_1 = ["10", "100", "200"]
        assert RetrievalMetricsCalculator.reciprocal_rank(expected, retrieved_1, k=20) == 1.0

        # Primer elemento relevante en pos 2 -> RR = 0.5
        retrieved_2 = ["100", "20", "200"]
        assert RetrievalMetricsCalculator.reciprocal_rank(expected, retrieved_2, k=20) == 0.5

        # Ningún elemento relevante -> RR = 0.0
        retrieved_none = ["100", "200", "300"]
        assert RetrievalMetricsCalculator.reciprocal_rank(expected, retrieved_none, k=20) == 0.0

    def test_average_precision(self):
        expected = ["10", "20"]

        # Ambas coincidencias en pos 1 y pos 2
        # P@1 = 1/1, P@2 = 2/2 -> AP = (1 + 1) / 2 = 1.0
        retrieved_perfect = ["10", "20", "30"]
        assert RetrievalMetricsCalculator.average_precision(expected, retrieved_perfect, k=20) == 1.0

        # Coincidencia en pos 2 (P@2 = 1/2)
        # AP = (0.5) / 2 = 0.25
        retrieved_partial = ["100", "10", "200"]
        assert RetrievalMetricsCalculator.average_precision(expected, retrieved_partial, k=20) == 0.25

    def test_recall_at_k(self):
        expected = ["10", "20", "30", "40", "50"]

        # 2 coincidencias de 5 -> Recall = 0.4
        retrieved = ["10", "20", "100", "200"]
        assert RetrievalMetricsCalculator.recall_at_k(expected, retrieved, k=20) == 0.4


class TestRetrievalEvaluator:
    @pytest.fixture
    def sample_csv_path(self, tmp_path: Path) -> Path:
        data = {
            "doc_id_1": [432, 2311],
            "doc_id_2": [6316, 747],
            "doc_id_3": [4760, 2706],
            "doc_id_4": [4302, 3511],
            "doc_id_5": [7360, 2545],
            "gemma": [
                json.dumps(["6236", "7473", "2575", "4302"]),
                json.dumps(["2706", "2021", "747", "2311"]),
            ],
            "qwen": [
                json.dumps(["2670", "432", "5624"]),
                json.dumps(["2706", "747", "2311", "3511", "2545"]),
            ],
        }
        df = pd.DataFrame(data)
        csv_file = tmp_path / "sample_questions.csv"
        df.to_csv(csv_file, index=False)
        return csv_file

    def test_run_evaluation(self, sample_csv_path: Path):
        evaluator = RetrievalEvaluator()
        summary_df, detailed_df = evaluator.run_evaluation(sample_csv_path, k=20)

        assert not summary_df.empty
        assert "Model" in summary_df.columns
        assert "MRR@20" in summary_df.columns
        assert "MAP@20" in summary_df.columns
        assert "Mean RECALL@20" in summary_df.columns

        assert "gemma_mrr" in detailed_df.columns
        assert "qwen_mrr" in detailed_df.columns
        assert len(detailed_df) == 2

    def test_format_results_table(self):
        summary_df = pd.DataFrame([
            {"Model": "Gemma", "MRR@20": 0.3327, "MAP@20": 0.1248, "Mean RECALL@20": 0.328},
            {"Model": "Qwen", "MRR@20": 0.4206, "MAP@20": 0.1538, "Mean RECALL@20": 0.3616},
        ])
        formatted = RetrievalEvaluator.format_results_table(summary_df)
        assert "| Model |" in formatted
        assert "| Gemma |" in formatted
        assert "| Qwen  |" in formatted
