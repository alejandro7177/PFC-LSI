import logging
from src.core.base import Pipeline
from src.repositories.base import BaseVectorRepository
from src.repositories.faiss_repository import FAISSRepository
from src.strategies.base import BaseLLMStrategy, BaseRetrievalStrategy
from src.strategies.embeddings import (
    DenseRetrievalStrategy,
    KeywordFilteredRetrievalStrategy,
    MockRetrievalStrategy,
)
from src.strategies.llm import Gemma12BStrategy, MockLLMStrategy
from src.filters.rag.query_preprocessor import QueryPreprocessorFilter
from src.filters.rag.retrieval_filter import RetrievalFilter
from src.filters.rag.prompt_builder import RAGPromptBuilderFilter
from src.filters.rag.generation_filter import GemmaGenerationFilter

logger = logging.getLogger(__name__)


class RAGPipelineFactory:
    """
    Fábrica para ensamblar tuberías RAG de recuperación y generación (Factory Pattern).
    Configura y conecta los filtros según la estrategia seleccionada.
    """

    @staticmethod
    def create_dense_rag_pipeline(
        llm_strategy: BaseLLMStrategy | None = None,
        repository: BaseVectorRepository | None = None,
        top_k: int = 5,
    ) -> Pipeline:
        """Construye una tubería RAG basada en búsqueda vectorial densa y Gemma 4 12B."""
        repo = repository or FAISSRepository()
        retrieval_strat = DenseRetrievalStrategy(repository=repo)
        llm_strat = llm_strategy or Gemma12BStrategy()

        pipeline = Pipeline()
        pipeline.add_filter(QueryPreprocessorFilter())
        pipeline.add_filter(RetrievalFilter(strategy=retrieval_strat, top_k=top_k))
        pipeline.add_filter(RAGPromptBuilderFilter())
        pipeline.add_filter(GemmaGenerationFilter(llm_strategy=llm_strat))

        logger.info("Tubería Dense RAG con Gemma 4 12B creada con éxito.")
        return pipeline

    @staticmethod
    def create_keyword_rag_pipeline(
        llm_strategy: BaseLLMStrategy | None = None,
        repository: BaseVectorRepository | None = None,
        keyword: str | None = None,
        top_k: int = 5,
    ) -> Pipeline:
        """Construye una tubería RAG basada en búsqueda delimitada por palabras clave y Gemma 4 12B."""
        repo = repository or FAISSRepository()
        retrieval_strat = KeywordFilteredRetrievalStrategy(repository=repo, keyword=keyword)
        llm_strat = llm_strategy or Gemma12BStrategy()

        pipeline = Pipeline()
        pipeline.add_filter(QueryPreprocessorFilter())
        pipeline.add_filter(RetrievalFilter(strategy=retrieval_strat, top_k=top_k))
        pipeline.add_filter(RAGPromptBuilderFilter())
        pipeline.add_filter(GemmaGenerationFilter(llm_strategy=llm_strat))

        logger.info("Tubería Keyword-Filtered RAG con Gemma 4 12B creada con éxito.")
        return pipeline

    @staticmethod
    def create_mock_rag_pipeline() -> Pipeline:
        """Construye una tubería RAG sintética/mock para pruebas unitarias sin dependencias externas."""
        retrieval_strat = MockRetrievalStrategy()
        llm_strat = MockLLMStrategy()

        pipeline = Pipeline()
        pipeline.add_filter(QueryPreprocessorFilter())
        pipeline.add_filter(RetrievalFilter(strategy=retrieval_strat, top_k=2))
        pipeline.add_filter(RAGPromptBuilderFilter())
        pipeline.add_filter(GemmaGenerationFilter(llm_strategy=llm_strat))

        logger.info("Tubería Mock RAG creada con éxito.")
        return pipeline

    @staticmethod
    def create_custom_rag_pipeline(
        retrieval_strategy: BaseRetrievalStrategy,
        llm_strategy: BaseLLMStrategy,
        top_k: int = 5,
        system_instruction: str | None = None,
    ) -> Pipeline:
        """Construye una tubería RAG personalizada con cualquier estrategia inyectada."""
        pipeline = Pipeline()
        pipeline.add_filter(QueryPreprocessorFilter())
        pipeline.add_filter(RetrievalFilter(strategy=retrieval_strategy, top_k=top_k))
        pipeline.add_filter(RAGPromptBuilderFilter(system_instruction=system_instruction))
        pipeline.add_filter(GemmaGenerationFilter(llm_strategy=llm_strategy))

        logger.info("Tubería RAG personalizada ensamblada con éxito.")
        return pipeline
