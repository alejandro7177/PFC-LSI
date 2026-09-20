import logging
from typing import Any
from src.core.config import config
from src.core.model_manager import model_manager
from src.strategies.base import BaseLLMStrategy

logger = logging.getLogger(__name__)


class Gemma12BStrategy(BaseLLMStrategy):
    """
    Estrategia de generación utilizando el modelo base Gemma 4 12B (google/gemma-4-12B-it).
    Utiliza el ModelManager Singleton para lazy loading y reutilización eficiente de memoria.
    """

    def __init__(
        self,
        model_name: str | None = None,
        cuda_gpu: str | None = None,
        torch_dtype: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        self._model_name = model_name or config.get("rag_system.llm.model_name", "google/gemma-4-12B-it")
        self.cuda_gpu = cuda_gpu or str(config.get("rag_system.llm.cuda_gpu", "0"))
        self.torch_dtype = torch_dtype or config.get("rag_system.llm.torch_dtype", "bfloat16")
        self.max_new_tokens = max_new_tokens or config.get("rag_system.llm.max_new_tokens", 512)
        self.temperature = temperature if temperature is not None else config.get("rag_system.llm.temperature", 0.2)
        self.top_p = top_p if top_p is not None else config.get("rag_system.llm.top_p", 0.95)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        model, tokenizer = model_manager.get_llm_and_tokenizer(
            model_name=self._model_name,
            cuda_gpu=self.cuda_gpu,
            torch_dtype=self.torch_dtype,
        )

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        if hasattr(tokenizer, "apply_chat_template"):
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        logger.info(f"Generando respuesta con Gemma 4 12B ({self._model_name})...")
        outputs = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_len:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

        return response_text.strip()


class MockLLMStrategy(BaseLLMStrategy):
    """
    Estrategia Mock de modelo de lenguaje para pruebas unitarias rápidas y evaluación de canalizaciones.
    """

    def __init__(self, model_name: str = "google/gemma-4-12B-it (Mock)"):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        logger.info(f"Generando respuesta sintética con {self._model_name}...")
        return (
            f"[Respuesta de Gemma 4 12B Mock]\n"
            f"Basado en el contexto analizado, los hallazgos principales responden a la consulta dada."
        )
