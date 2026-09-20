from abc import ABC, abstractmethod
from src.core.dto import RAGQuery, RetrievedDocument


class BaseLLMStrategy(ABC):
    """Interfaz abstracta para estrategias de modelos de lenguaje LLM (Strategy Pattern)."""

    @abstractmethod
    def generate(self, prompt: str, system_instruction: str = "") -> str:
        """Genera una respuesta textual dada la instrucción o prompt recibido."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Devuelve el nombre/identificador del modelo base."""
        pass


class BaseRetrievalStrategy(ABC):
    """Interfaz abstracta para estrategias de recuperación de información (Strategy Pattern)."""

    @abstractmethod
    def retrieve(self, query: RAGQuery, top_k: int = 5) -> list[RetrievedDocument]:
        """Recupera los K documentos más relevantes para una consulta dada."""
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Devuelve el nombre identificador de la estrategia de retrieval."""
        pass
