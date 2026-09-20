from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGQuery:
    """Representa la consulta de usuario procesada fluyendo por el pipeline."""

    raw_query: str
    normalized_query: str = ""
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.normalized_query:
            self.normalized_query = self.raw_query.strip()


@dataclass
class RetrievedDocument:
    """Representa un documento o pasaje científico recuperado por la estrategia de retrieval."""

    doc_id: str
    abstract: str
    score: float = 0.0
    keywords: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGContext:
    """Encapsula la consulta original y los documentos recuperados."""

    query: RAGQuery
    documents: list[RetrievedDocument] = field(default_factory=list)
    strategy_used: str = ""


@dataclass
class RAGFormattedPrompt:
    """Encapsula el prompt final estructurado para el modelo de lenguaje (Gemma 4 12B)."""

    query: RAGQuery
    prompt_text: str
    documents: list[RetrievedDocument] = field(default_factory=list)
    system_instruction: str = ""


@dataclass
class RAGResponse:
    """Respuesta final sintetizada producida por el sistema RAG."""

    query: str
    answer: str
    source_documents: list[RetrievedDocument] = field(default_factory=list)
    model_name: str = ""
    retrieval_strategy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
