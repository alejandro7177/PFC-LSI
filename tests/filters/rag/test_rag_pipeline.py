import pytest
from src.core.dto import RAGContext, RAGFormattedPrompt, RAGQuery, RAGResponse, RetrievedDocument
from src.core.factory import RAGPipelineFactory
from src.filters.rag.generation_filter import GemmaGenerationFilter
from src.filters.rag.prompt_builder import RAGPromptBuilderFilter
from src.filters.rag.query_preprocessor import QueryPreprocessorFilter
from src.filters.rag.retrieval_filter import RetrievalFilter
from src.strategies.embeddings import MockRetrievalStrategy
from src.strategies.llm import MockLLMStrategy


class TestRAGPipelineFilters:
    def test_query_preprocessor_filter_string_input(self):
        filter_step = QueryPreprocessorFilter()
        stream = (item for item in ["  What is atmospheric physics?  "])
        results = list(filter_step.process(stream))

        assert len(results) == 1
        assert isinstance(results[0], RAGQuery)
        assert results[0].raw_query == "What is atmospheric physics?"
        assert results[0].normalized_query == "What is atmospheric physics?"

    def test_query_preprocessor_filter_dict_input(self):
        filter_step = QueryPreprocessorFilter()
        stream = (item for item in [{"question": "Climate models", "keywords": "climate, physics", "extra": "data"}])
        results = list(filter_step.process(stream))

        assert len(results) == 1
        assert results[0].raw_query == "Climate models"
        assert results[0].keywords == ["climate", "physics"]
        assert results[0].metadata["extra"] == "data"

    def test_retrieval_filter(self):
        mock_docs = [
            RetrievedDocument(doc_id="doc_1", abstract="Climate dynamics summary", score=0.9),
        ]
        strat = MockRetrievalStrategy(mock_documents=mock_docs)
        filter_step = RetrievalFilter(strategy=strat, top_k=1)

        query_obj = RAGQuery(raw_query="climate")
        stream = (item for item in [query_obj])
        results = list(filter_step.process(stream))

        assert len(results) == 1
        assert isinstance(results[0], RAGContext)
        assert results[0].query == query_obj
        assert len(results[0].documents) == 1
        assert results[0].documents[0].doc_id == "doc_1"

    def test_prompt_builder_filter(self):
        filter_step = RAGPromptBuilderFilter(system_instruction="Test system instruction")
        ctx = RAGContext(
            query=RAGQuery(raw_query="Explain climate dynamics"),
            documents=[
                RetrievedDocument(doc_id="doc_1", abstract="Abstract content 1", keywords="climate"),
            ],
        )
        stream = (item for item in [ctx])
        results = list(filter_step.process(stream))

        assert len(results) == 1
        assert isinstance(results[0], RAGFormattedPrompt)
        assert "Abstract content 1" in results[0].prompt_text
        assert "Explain climate dynamics" in results[0].prompt_text
        assert results[0].system_instruction == "Test system instruction"

    def test_gemma_generation_filter(self):
        mock_llm = MockLLMStrategy(model_name="google/gemma-4-12B-it")
        filter_step = GemmaGenerationFilter(llm_strategy=mock_llm)

        formatted_prompt = RAGFormattedPrompt(
            query=RAGQuery(raw_query="What is physics?"),
            prompt_text="Formatted prompt text",
            documents=[RetrievedDocument(doc_id="doc_1", abstract="Abstract 1")],
            system_instruction="System instruction",
        )
        stream = (item for item in [formatted_prompt])
        results = list(filter_step.process(stream))

        assert len(results) == 1
        assert isinstance(results[0], RAGResponse)
        assert results[0].query == "What is physics?"
        assert results[0].model_name == "google/gemma-4-12B-it"
        assert "[Respuesta de Gemma 4 12B Mock]" in results[0].answer
        assert len(results[0].source_documents) == 1

    def test_end_to_end_mock_pipeline(self):
        pipeline = RAGPipelineFactory.create_mock_rag_pipeline()
        input_stream = (item for item in ["¿Cómo afecta la radiación solar al clima?"])
        results = list(pipeline.run(input_stream))

        assert len(results) == 1
        response = results[0]
        assert isinstance(response, RAGResponse)
        assert response.query == "¿Cómo afecta la radiación solar al clima?"
        assert len(response.source_documents) > 0
        assert "gemma-4-12b-it" in response.model_name.lower() or "gemma 4 12b" in response.model_name.lower()
