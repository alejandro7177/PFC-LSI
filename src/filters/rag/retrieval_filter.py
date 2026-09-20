import logging
from typing import Generator
from src.core.base import Filter
from src.core.config import config
from src.core.dto import RAGContext, RAGQuery
from src.strategies.base import BaseRetrievalStrategy
from src.strategies.embeddings import DenseRetrievalStrategy

logger = logging.getLogger(__name__)


class RetrievalFilter(Filter):
    """
    Filtro de recuperación de documentos (Pipeline step 2).
    Recibe un stream de RAGQuery, utiliza una RetrievalStrategy desacoplada
    y emite un stream de RAGContext.
    """

    def __init__(self, strategy: BaseRetrievalStrategy | None = None, top_k: int | None = None):
        self.strategy = strategy or DenseRetrievalStrategy()
        self.top_k = top_k or config.get("rag_system.retrieval.top_k", 5)

    def process(self, stream: Generator[RAGQuery, None, None]) -> Generator[RAGContext, None, None]:
        for query_obj in stream:
            documents = self.strategy.retrieve(query=query_obj, top_k=self.top_k)
            logger.debug(
                f"RetrievalFilter: Recuperados {len(documents)} documentos para consulta '{query_obj.normalized_query[:40]}...'"
            )
            yield RAGContext(
                query=query_obj,
                documents=documents,
                strategy_used=self.strategy.strategy_name,
            )
