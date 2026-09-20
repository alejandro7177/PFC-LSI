from unittest.mock import MagicMock, patch
import pytest
from src.core.dto import RAGQuery, RetrievedDocument
from src.core.model_manager import ModelManager
from src.repositories.base import BaseVectorRepository
from src.strategies.embeddings import DenseRetrievalStrategy, KeywordFilteredRetrievalStrategy
from src.strategies.llm import Gemma12BStrategy, MockLLMStrategy


class TestLLMAndRetrievalStrategies:
    def test_mock_llm_strategy(self):
        strategy = MockLLMStrategy(model_name="google/gemma-4-12B-it")
        assert strategy.model_name == "google/gemma-4-12B-it"

        answer = strategy.generate(prompt="What is atmospheric dynamics?", system_instruction="System")
        assert "[Respuesta de Gemma 4 12B Mock]" in answer

    @patch("src.core.model_manager.model_manager.get_llm_and_tokenizer")
    def test_gemma_12b_strategy(self, mock_get_llm):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_tokenizer.apply_chat_template.return_value = "Formatted prompt"
        mock_tokenizer.return_value.to.return_value = {"input_ids": MagicMock(shape=(1, 5))}
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "Respuesta generada por Gemma 4 12B real."

        mock_model.device = "cpu"
        mock_model.generate.return_value = [[0, 1, 2, 3, 4, 5, 6]]
        mock_get_llm.return_value = (mock_model, mock_tokenizer)

        strategy = Gemma12BStrategy(model_name="google/gemma-4-12B-it", cuda_gpu="0")
        assert strategy.model_name == "google/gemma-4-12B-it"

        output = strategy.generate(prompt="Explain RAG", system_instruction="System prompt")
        assert output == "Respuesta generada por Gemma 4 12B real."
        mock_model.generate.assert_called_once()

    def test_model_manager_singleton(self):
        m1 = ModelManager()
        m2 = ModelManager()
        assert m1 is m2

    def test_dense_retrieval_strategy(self):
        mock_repo = MagicMock(spec=BaseVectorRepository)
        mock_repo.search.return_value = [
            RetrievedDocument(doc_id="d1", abstract="Abs 1", score=0.9),
        ]

        strat = DenseRetrievalStrategy(repository=mock_repo)
        assert strat.strategy_name == "DenseRetrievalStrategy"

        query = RAGQuery(raw_query="physics")
        docs = strat.retrieve(query=query, top_k=1)
        assert len(docs) == 1
        assert docs[0].doc_id == "d1"
        mock_repo.search.assert_called_once_with(query="physics", k=1)

    def test_keyword_filtered_retrieval_strategy(self):
        mock_repo = MagicMock(spec=BaseVectorRepository)
        mock_repo.search_by_keyword.return_value = [
            RetrievedDocument(doc_id="dk1", abstract="Abs K1", score=0.95),
        ]

        strat = KeywordFilteredRetrievalStrategy(repository=mock_repo, keyword="climate")
        assert strat.strategy_name == "KeywordFilteredRetrievalStrategy"

        query = RAGQuery(raw_query="climate change")
        docs = strat.retrieve(query=query, top_k=2)
        assert len(docs) == 1
        assert docs[0].doc_id == "dk1"
        mock_repo.search_by_keyword.assert_called_once_with(keyword="climate", query="climate change", k=2)
