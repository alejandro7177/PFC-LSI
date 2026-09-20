from abc import ABC, abstractmethod
from typing import Any
from src.core.dto import RetrievedDocument


class BaseVectorRepository(ABC):
    """Interfaz abstracta para repositorios de almacén vectorial y metadatos (Repository Pattern)."""

    @abstractmethod
    def search(self, query: str, k: int | None = None) -> list[RetrievedDocument]:
        """Realiza una búsqueda de k documentos más similares a la consulta."""
        pass

    @abstractmethod
    def search_by_keyword(
        self,
        keyword: str,
        query: str | None = None,
        k: int | None = None
    ) -> list[RetrievedDocument]:
        """Realiza una búsqueda de k documentos delimitados por una palabra clave."""
        pass

    @abstractmethod
    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Añade documentos al almacén vectorial."""
        pass
