import logging
from typing import Generator
from src.core.base import Filter
from src.core.dto import RAGFormattedPrompt, RAGResponse
from src.strategies.base import BaseLLMStrategy
from src.strategies.llm import Gemma12BStrategy

logger = logging.getLogger(__name__)


class GemmaGenerationFilter(Filter):
    """
    Filtro de generación LLM con Gemma 4 12B (Pipeline step 4).
    Recibe el stream de RAGFormattedPrompt, ejecuta la inferencia del LLM
    vía la estrategia inyectada y emite el objeto RAGResponse.
    """

    def __init__(self, llm_strategy: BaseLLMStrategy | None = None):
        self.llm_strategy = llm_strategy or Gemma12BStrategy()

    def process(self, stream: Generator[RAGFormattedPrompt, None, None]) -> Generator[RAGResponse, None, None]:
        for formatted_prompt in stream:
            logger.info(
                f"GemmaGenerationFilter: Generando respuesta usando {self.llm_strategy.model_name}..."
            )
            answer_text = self.llm_strategy.generate(
                prompt=formatted_prompt.prompt_text,
                system_instruction=formatted_prompt.system_instruction,
            )

            yield RAGResponse(
                query=formatted_prompt.query.raw_query,
                answer=answer_text,
                source_documents=formatted_prompt.documents,
                model_name=self.llm_strategy.model_name,
                metadata={
                    "query_keywords": formatted_prompt.query.keywords,
                    "num_sources": len(formatted_prompt.documents),
                },
            )
