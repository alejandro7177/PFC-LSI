import logging
from src.core.dto import RAGQuery, RetrievedDocument
from src.repositories.base import BaseVectorRepository
from src.repositories.faiss_repository import FAISSRepository
from src.strategies.base import BaseRetrievalStrategy

logger = logging.getLogger(__name__)


class DenseRetrievalStrategy(BaseRetrievalStrategy):
    """
    Estrategia de búsqueda densa utilizando vectores y el repositorio FAISSRepository.
    """

    def __init__(self, repository: BaseVectorRepository | None = None):
        self.repository = repository or FAISSRepository()

    @property
    def strategy_name(self) -> str:
        return "DenseRetrievalStrategy"

    def retrieve(self, query: RAGQuery, top_k: int = 5) -> list[RetrievedDocument]:
        search_query = query.normalized_query or query.raw_query
        logger.info(f"Ejecutando {self.strategy_name} para consulta: '{search_query[:50]}...' (top_k={top_k})")

        raw_results = self.repository.search(query=search_query, k=top_k)

        documents = []
        for item in raw_results:
            if isinstance(item, RetrievedDocument):
                documents.append(item)
            elif hasattr(item, "to_dto"):
                documents.append(item.to_dto())
            else:
                documents.append(
                    RetrievedDocument(
                        doc_id=str(getattr(item, "doc_id", "")),
                        score=float(getattr(item, "score", 0.0)),
                        abstract=str(getattr(item, "abstract", "")),
                        keywords=str(getattr(item, "keywords", "")),
                    )
                )

        return documents


class KeywordFilteredRetrievalStrategy(BaseRetrievalStrategy):
    """
    Estrategia de búsqueda híbrida/delimitada por palabras clave usando el mapeo del repositorio.
    """

    def __init__(self, repository: BaseVectorRepository | None = None, keyword: str | None = None):
        self.repository = repository or FAISSRepository()
        self.keyword = keyword

    @property
    def strategy_name(self) -> str:
        return "KeywordFilteredRetrievalStrategy"

    def retrieve(self, query: RAGQuery, top_k: int = 5) -> list[RetrievedDocument]:
        target_kw = self.keyword
        if not target_kw and query.keywords:
            target_kw = query.keywords[0]

        search_query = query.normalized_query or query.raw_query

        if target_kw:
            logger.info(f"Ejecutando {self.strategy_name} para keyword '{target_kw}'...")
            try:
                raw_results = self.repository.search_by_keyword(keyword=target_kw, query=search_query, k=top_k)
            except (KeyError, FileNotFoundError, ValueError) as e:
                logger.warning(f"Fallback a búsqueda densa estándar debido a: {e}")
                raw_results = self.repository.search(query=search_query, k=top_k)
        else:
            logger.info(f"Sin keyword disponible. Fallback a búsqueda densa estándar.")
            raw_results = self.repository.search(query=search_query, k=top_k)

        documents = []
        for item in raw_results:
            if isinstance(item, RetrievedDocument):
                documents.append(item)
            elif hasattr(item, "to_dto"):
                documents.append(item.to_dto())
            else:
                documents.append(
                    RetrievedDocument(
                        doc_id=str(getattr(item, "doc_id", "")),
                        score=float(getattr(item, "score", 0.0)),
                        abstract=str(getattr(item, "abstract", "")),
                        keywords=str(getattr(item, "keywords", "")),
                    )
                )

        return documents


class MockRetrievalStrategy(BaseRetrievalStrategy):
    """
    Estrategia Mock de recuperación para pruebas unitarias sin dependencia de FAISS ni embeddings.
    """

    def __init__(self, mock_documents: list[RetrievedDocument] | None = None):
        self._mock_documents = mock_documents or [
            RetrievedDocument(
                doc_id="doc_mock_1",
                abstract="Este es un resumen científico mock sobre física de la atmósfera y cambio climático.",
                score=0.98,
                keywords="atmospheric physics, climate",
            ),
            RetrievedDocument(
                doc_id="doc_mock_2",
                abstract="Estudio experimental de la dinámica de fluidos geofísicos.",
                score=0.85,
                keywords="geophysical fluids, dynamics",
            ),
        ]

    @property
    def strategy_name(self) -> str:
        return "MockRetrievalStrategy"

    def retrieve(self, query: RAGQuery, top_k: int = 5) -> list[RetrievedDocument]:
        logger.info(f"Recuperando {min(top_k, len(self._mock_documents))} documentos mock.")
        return self._mock_documents[:top_k]
