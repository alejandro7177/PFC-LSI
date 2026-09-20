import logging
from typing import Generator
from src.core.base import Filter
from src.core.config import config
from src.core.dto import RAGContext, RAGFormattedPrompt

logger = logging.getLogger(__name__)


class RAGPromptBuilderFilter(Filter):
    """
    Filtro de construcción de prompts para Gemma 4 12B (Pipeline step 3).
    Formatea el contexto recuperado y la consulta de usuario en una estructura
    de instrucción optimizada para el modelo Gemma 4.
    """

    def __init__(self, system_instruction: str | None = None):
        self.system_instruction = system_instruction or config.get(
            "rag_system.prompt.system_instruction",
            "Eres un asistente de investigación altamente preciso especializado en resumir y responder preguntas basadas exclusivamente en el contexto científico proporcionado.",
        )

    def process(self, stream: Generator[RAGContext, None, None]) -> Generator[RAGFormattedPrompt, None, None]:
        for context in stream:
            formatted_docs = []
            for idx, doc in enumerate(context.documents, start=1):
                doc_str = f"[Documento {idx} (ID: {doc.doc_id})]\n"
                if doc.keywords:
                    doc_str += f"Palabras Clave: {doc.keywords}\n"
                doc_str += f"Resumen: {doc.abstract.strip()}"
                formatted_docs.append(doc_str)

            context_str = "\n\n".join(formatted_docs) if formatted_docs else "No se encontraron documentos de contexto."

            prompt_text = (
                f"CONTEXTO CIENTÍFICO PROPORCIONADO:\n"
                f"----------------------------------------\n"
                f"{context_str}\n"
                f"----------------------------------------\n\n"
                f"PREGUNTA DEL USUARIO: {context.query.raw_query}\n\n"
                f"INSTRUCCIONES:\n"
                f"Responde a la pregunta del usuario utilizando de forma estricta y detallada la información dada en el contexto. "
                f"Si la información no se encuentra en el contexto, indícalo claramente."
            )

            logger.debug(f"RAGPromptBuilderFilter: Prompt construido para consulta '{context.query.normalized_query[:40]}...'")

            yield RAGFormattedPrompt(
                query=context.query,
                prompt_text=prompt_text,
                documents=context.documents,
                system_instruction=self.system_instruction,
            )
