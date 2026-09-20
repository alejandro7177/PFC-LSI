import logging
from typing import Any, Generator
from src.core.base import Filter
from src.core.dto import RAGQuery

logger = logging.getLogger(__name__)


class QueryPreprocessorFilter(Filter):
    """
    Filtro de preprocesamiento de la consulta (Pipeline step 1).
    Convierte entradas del stream (str, dict, RAGQuery) en objetos RAGQuery tipados,
    normalizando el texto y extrayendo metadatos.
    """

    def process(self, stream: Generator[Any, None, None]) -> Generator[RAGQuery, None, None]:
        for item in stream:
            if isinstance(item, RAGQuery):
                query_obj = item
            elif isinstance(item, str):
                raw_text = item.strip()
                query_obj = RAGQuery(raw_query=raw_text, normalized_query=raw_text)
            elif isinstance(item, dict):
                raw_text = str(item.get("query", item.get("raw_query", item.get("question", "")))).strip()
                kws = item.get("keywords", [])
                if isinstance(kws, str):
                    kws = [k.strip() for k in kws.split(",") if k.strip()]
                meta = {k: v for k, v in item.items() if k not in ("query", "raw_query", "question", "keywords")}
                query_obj = RAGQuery(raw_query=raw_text, normalized_query=raw_text, keywords=kws, metadata=meta)
            else:
                raw_text = str(item).strip()
                query_obj = RAGQuery(raw_query=raw_text, normalized_query=raw_text)

            logger.debug(f"QueryPreprocessorFilter: Procesada consulta '{query_obj.normalized_query[:40]}...'")
            yield query_obj
