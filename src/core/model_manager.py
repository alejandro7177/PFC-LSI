import logging
import threading
from typing import Any, Tuple
import torch

from src.core.config import config

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Gestor Singleton para la carga diferida (lazy loading), gestión de memoria VRAM/RAM
    y reutilización eficiente de modelos pesados (LLM Gemma 4 12B y Embeddings).
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._models = {}
                    cls._instance._tokenizers = {}
        return cls._instance

    def get_llm_and_tokenizer(
        self,
        model_name: str | None = None,
        cuda_gpu: str | None = None,
        torch_dtype: str | None = None,
    ) -> Tuple[Any, Any]:
        """
        Obtiene o carga en memoria el modelo de lenguaje (Gemma 4 12B) y su tokenizer.
        """
        model_name = model_name or config.get("rag_system.llm.model_name", "google/gemma-4-12B-it")
        gpu_id = cuda_gpu or str(config.get("rag_system.llm.cuda_gpu", "0"))
        dtype_str = torch_dtype or config.get("rag_system.llm.torch_dtype", "bfloat16")

        cache_key = f"llm_{model_name}"

        with self._lock:
            if cache_key in self._models:
                return self._models[cache_key], self._tokenizers[cache_key]

            logger.info(f"Cargando modelo Gemma 4 12B '{model_name}' en GPU:{gpu_id} ({dtype_str})...")

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                raise ImportError(
                    "El paquete 'transformers' no está instalado. Instálalo con 'pip install transformers torch'."
                )

            device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

            dtype = torch.bfloat16
            if dtype_str == "float16":
                dtype = torch.float16
            elif dtype_str == "float32":
                dtype = torch.float32

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device if device.startswith("cuda") else "cpu",
                trust_remote_code=True,
            )

            self._models[cache_key] = model
            self._tokenizers[cache_key] = tokenizer

            logger.info(f"Modelo LLM '{model_name}' cargado con éxito en {device}.")
            return model, tokenizer

    def clear_cache(self):
        """Limpia el caché de modelos y libera memoria GPU."""
        with self._lock:
            self._models.clear()
            self._tokenizers.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Caché de modelos limpiado y memoria VRAM liberada.")


# Instancia global del Singleton
model_manager = ModelManager()
