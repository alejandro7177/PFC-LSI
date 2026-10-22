"""
Evaluador de búsqueda vectorial para Gemma y Qwen.

Carga el CSV con preguntas (columnas: ids_docs, question, specific),
ejecuta búsqueda vectorial densa top-k en ambos vector stores (Gemma y Qwen),
y guarda los doc_ids recuperados como nuevas columnas `gemma` y `qwen` en el CSV de salida.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.core.config import config
from src.repositories.faiss_repository import FAISSRepository

logger = logging.getLogger(__name__)


class VectorSearchEvaluator:
    """
    Evaluador de búsqueda vectorial sobre los modelos Gemma y Qwen.

    Para cada pregunta del CSV de entrada, realiza una búsqueda densa top-k
    en ambos vector stores y añade los resultados como columnas `gemma` y `qwen`
    (listas JSON de doc_ids) al CSV de salida.
    """

    def __init__(
        self,
        qwen_store_dir: str | Path | None = None,
        gemma_store_dir: str | Path | None = None,
        top_k: int = 40,
        use_gpu: bool | None = None,
    ):
        """
        Inicializa el evaluador cargando ambos vector stores FAISS.

        Args:
            qwen_store_dir: Directorio del vector store de Qwen (por defecto: config).
            gemma_store_dir: Directorio del vector store de Gemma (por defecto: config).
            top_k: Número de documentos a recuperar por cada búsqueda.
            use_gpu: Si se usa GPU para los embeddings (por defecto: config).
        """
        self.top_k = top_k

        qwen_dir = Path(qwen_store_dir or config.get("faiss_repository.vector_store_dir", "vectorStores/Qwen-8B"))
        gemma_dir = Path(gemma_store_dir or config.get("faiss_repository.gemma_vector_store_dir", "vectorStores/gemma4"))

        gpu_flag = use_gpu if use_gpu is not None else config.get("faiss_repository.use_gpu", True)

        # ---- Repositorio Qwen ----
        qwen_model = config.get("faiss_repository.model_name", "Qwen/Qwen3-Embedding-8B")
        logger.info(f"Inicializando repositorio Qwen ({qwen_model}) desde '{qwen_dir}'...")
        self.qwen_repo = FAISSRepository(model_name=qwen_model, use_gpu=gpu_flag)
        self.qwen_repo.load(directory=qwen_dir)
        logger.info(f"  → {self.qwen_repo.index.ntotal} documentos cargados en Qwen.")

        # ---- Repositorio Gemma ----
        gemma_model = config.get("faiss_repository.gemma_model_name", "google/embeddinggemma-300m")
        logger.info(f"Inicializando repositorio Gemma ({gemma_model}) desde '{gemma_dir}'...")
        self.gemma_repo = FAISSRepository(model_name=gemma_model, use_gpu=gpu_flag)
        self.gemma_repo.load(directory=gemma_dir)
        logger.info(f"  → {self.gemma_repo.index.ntotal} documentos cargados en Gemma.")

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    @staticmethod
    def _search_to_json(results) -> str:
        """Convierte la lista de SearchResult a JSON array de doc_ids (strings)."""
        doc_ids = [str(r.doc_id) for r in results]
        return json.dumps(doc_ids)

    # ------------------------------------------------------------------
    # Método principal de evaluación
    # ------------------------------------------------------------------

    def evaluate(
        self,
        csv_path: str | Path,
        output_path: str | Path,
        top_k: int | None = None,
        question_col: str = "question",
        overwrite_cols: bool = False,
    ) -> pd.DataFrame:
        """
        Evalúa búsqueda vectorial para cada pregunta del CSV de entrada.

        Para cada fila:
          - Ejecuta búsqueda top-k en el repositorio Qwen → columna ``qwen``
          - Ejecuta búsqueda top-k en el repositorio Gemma → columna ``gemma``

        Args:
            csv_path: Ruta al CSV de entrada (debe tener columna `question`).
            output_path: Ruta donde guardar el CSV de salida enriquecido.
            top_k: Número de documentos a recuperar (sobreescribe el valor del constructor).
            question_col: Nombre de la columna con el texto de la pregunta.
            overwrite_cols: Si es True, sobreescribe las columnas `gemma`/`qwen` si ya existen.

        Returns:
            DataFrame con las columnas `gemma` y `qwen` añadidas.

        Raises:
            FileNotFoundError: Si el CSV de entrada no existe.
            ValueError: Si la columna de preguntas no existe en el CSV.
        """
        csv_path = Path(csv_path)
        output_path = Path(output_path)
        k = top_k if top_k is not None else self.top_k

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV de entrada no encontrado: '{csv_path}'")

        logger.info(f"Cargando dataset desde '{csv_path}'...")
        df = pd.read_csv(csv_path)

        if question_col not in df.columns:
            raise ValueError(
                f"La columna '{question_col}' no existe en el CSV. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        # Verificar si ya existen las columnas de resultados
        for col in ("gemma", "qwen"):
            if col in df.columns and not overwrite_cols:
                logger.warning(
                    f"La columna '{col}' ya existe en el CSV. "
                    "Use overwrite_cols=True para sobrescribirla."
                )

        total = len(df)
        logger.info(f"Iniciando búsqueda vectorial top-{k} para {total} preguntas...")

        qwen_results: list[str] = []
        gemma_results: list[str] = []

        for idx, row in df.iterrows():
            question = str(row[question_col]).strip()

            if not question:
                logger.warning(f"Fila {idx}: pregunta vacía, omitiendo.")
                qwen_results.append(json.dumps([]))
                gemma_results.append(json.dumps([]))
                continue

            # Búsqueda Qwen
            try:
                qwen_hits = self.qwen_repo.search(query=question, k=k)
                qwen_results.append(self._search_to_json(qwen_hits))
            except Exception as e:
                logger.error(f"Fila {idx} — Error en búsqueda Qwen: {e}")
                qwen_results.append(json.dumps([]))

            # Búsqueda Gemma
            try:
                gemma_hits = self.gemma_repo.search(query=question, k=k)
                gemma_results.append(self._search_to_json(gemma_hits))
            except Exception as e:
                logger.error(f"Fila {idx} — Error en búsqueda Gemma: {e}")
                gemma_results.append(json.dumps([]))

            # Log de progreso cada 50 preguntas
            row_num = int(idx) + 1  # type: ignore[arg-type]
            if row_num % 50 == 0 or row_num == total:
                logger.info(f"  Progreso: {row_num}/{total} preguntas procesadas.")

        # Añadir/sobreescribir columnas en el DataFrame
        df["qwen"] = qwen_results
        df["gemma"] = gemma_results

        # Guardar resultado
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"CSV de evaluación guardado en: '{output_path}' ({total} filas).")

        return df
