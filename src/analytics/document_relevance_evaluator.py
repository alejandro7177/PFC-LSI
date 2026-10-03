import json
import logging
from pathlib import Path
import re
from typing import Any, Sequence
import numpy as np
import pandas as pd

from src.core.config import config
from src.strategies.base import BaseLLMStrategy
from src.strategies.llm import Gemma12BStrategy, MockLLMStrategy

logger = logging.getLogger(__name__)


class DocumentRelevanceEvaluator:
    """
    Evaluador Híbrido de Suficiencia y Rendimiento de Recuperación de Documentos (Gemma vs Qwen).
    Soporta:
    - Métricas matemáticas deterministas (Recall@20, Hit Rate, MRR).
    - Evaluación cualitativa con LLM Judge (Suficiencia de Contexto mediante Prompt).
    - Evaluación Híbrida Combinada.
    """

    def __init__(
        self,
        llm_strategy: BaseLLMStrategy | None = None,
        prompt_path: str | Path | None = None,
        doc_id_cols: Sequence[str] | None = None,
        top_k: int = 20,
    ):
        self.llm_strategy = llm_strategy or MockLLMStrategy()
        if prompt_path:
            self.prompt_path = Path(prompt_path)
        else:
            conf_path = config.get("doc_relevance_cv.prompt_path", "prompts/context_sufficiency.md")
            self.prompt_path = Path(conf_path)
        self._prompt_template = self._load_prompt_template()
        self.doc_id_cols = list(doc_id_cols) if doc_id_cols else [f"doc_id_{i}" for i in range(1, 6)]
        self.top_k = top_k

    def _load_prompt_template(self) -> str:
        """Carga la plantilla de prompt desde la ruta configurada o usa una por defecto."""
        if not self.prompt_path.exists():
            logger.warning(
                f"No se encontró el archivo de prompt en '{self.prompt_path}'. Usando plantilla por defecto."
            )
            return (
                "You are an expert scientific literature evaluator.\n"
                "Evaluate if the provided retrieved document abstracts contain sufficient information to answer the question.\n"
                "Assign a sufficiency score from 0.0 to 1.0.\n"
                "Return ONLY a JSON object: {\"sufficiency_score\": 0.85, \"reasoning\": \"...\"}\n\n"
                "QUESTION:\n{question}\n\nRETRIEVED ABSTRACTS:\n{abstracts}\n"
            )
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def parse_retrieved_list(val: Any) -> list[str]:
        """Parsea la cadena JSON o lista de documentos recuperados a una lista de strings."""
        if isinstance(val, list):
            return [str(item) for item in val]
        if pd.isna(val) or not str(val).strip():
            return []
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    def extract_expected_ids(self, row: pd.Series) -> list[str]:
        """Extrae los IDs esperados de documentos relevantes de una fila del DataFrame."""
        expected_ids = []
        for col in self.doc_id_cols:
            if col in row and pd.notna(row[col]):
                val = row[col]
                try:
                    expected_ids.append(str(int(val)))
                except (ValueError, TypeError):
                    expected_ids.append(str(val).strip())
        return expected_ids

    @staticmethod
    def compute_mrr(expected_ids: list[str], retrieved_ids: list[str], top_k: int = 20) -> float:
        """Calcula el Reciprocal Rank (1 / rank) del primer documento relevante recuperado."""
        if not expected_ids:
            return 0.0
        expected_set = set(expected_ids)
        for rank, doc_id in enumerate(retrieved_ids[:top_k], start=1):
            if doc_id in expected_set:
                return round(1.0 / rank, 4)
        return 0.0

    def parse_llm_response(self, response_text: str) -> dict[str, Any]:
        """Parsea la respuesta JSON/texto del LLM Judge extrayendo 'sufficiency_score' y 'reasoning'."""
        if not response_text or not isinstance(response_text, str):
            return {"sufficiency_score": 0.0, "reasoning": "Respuesta vacía o inválida del modelo."}

        cleaned = response_text.strip()
        if "```" in cleaned:
            cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                score_val = data.get("sufficiency_score", data.get("relevance_score", 0.0))
                score = float(score_val)
                reason = str(data.get("reasoning", "Sin justificación proporcionada.")).strip()
                return {
                    "sufficiency_score": float(np.clip(score, 0.0, 1.0)),
                    "reasoning": reason,
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        score_match = re.search(r'"(?:sufficiency_score|relevance_score)"\s*:\s*([0-9]+(?:\.[0-9]+)?)', cleaned, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                return {
                    "sufficiency_score": float(np.clip(score, 0.0, 1.0)),
                    "reasoning": f"Extraído por regex de: {cleaned[:100]}",
                }
            except ValueError:
                pass

        float_matches = re.findall(r"\b0\.\d+|\b1\.0|\b0\b|\b1\b", cleaned)
        if float_matches:
            try:
                score = float(float_matches[0])
                return {
                    "sufficiency_score": float(np.clip(score, 0.0, 1.0)),
                    "reasoning": f"Float inferido de texto: {cleaned[:100]}",
                }
            except ValueError:
                pass

        return {
            "sufficiency_score": 0.5,
            "reasoning": f"No se pudo parsear el puntaje. Texto original: {cleaned[:100]}",
        }

    def evaluate_llm_context_sufficiency(self, question: str, abstracts_text: str) -> dict[str, Any]:
        """Evalúa cualitativamente mediante LLM si el texto de resúmenes recuperados responde la pregunta."""
        if not question or not abstracts_text:
            return {"sufficiency_score": 0.0, "reasoning": "Pregunta o resúmenes vacíos."}

        prompt = (
            self._prompt_template
            .replace("{{question}}", str(question).strip())
            .replace("{{abstracts}}", str(abstracts_text).strip())
            .replace("{{abstract}}", str(abstracts_text).strip())
            .replace("{question}", str(question).strip())
            .replace("{abstracts}", str(abstracts_text).strip())
            .replace("{abstract}", str(abstracts_text).strip())
        )

        response = self.llm_strategy.generate(prompt=prompt)
        return self.parse_llm_response(response)

    def extract_abstracts_for_row(self, row: pd.Series, retrieved_ids: list[str]) -> str:
        """Recolecta el texto de los resúmenes asociados a los documentos recuperados o disponibles en la fila."""
        abstracts = []
        retrieved_set = set(retrieved_ids)
        for i in range(1, 6):
            doc_col = f"doc_id_{i}"
            abs_col = f"abstract_{i}"
            if doc_col in row and abs_col in row and pd.notna(row[abs_col]):
                val = row[doc_col]
                try:
                    doc_id = str(int(val))
                except (ValueError, TypeError):
                    doc_id = str(val).strip()
                if not retrieved_set or doc_id in retrieved_set:
                    abstracts.append(f"Documento #{doc_id}:\n{str(row[abs_col]).strip()}")
        if not abstracts:
            for i in range(1, 6):
                abs_col = f"abstract_{i}"
                if abs_col in row and pd.notna(row[abs_col]):
                    abstracts.append(f"Documento candidato {i}:\n{str(row[abs_col]).strip()}")
        return "\n\n".join(abstracts)

    def evaluate_hybrid_sufficiency(
        self,
        df: pd.DataFrame,
        eval_mode: str = "hybrid",
        top_k: int = 20,
        sample_limit: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evalúa la suficiencia de recuperación combinando métricas matemáticas y evaluación cualitativa LLM Judge.

        Parámetros:
            - eval_mode: 'math', 'llm', o 'hybrid'
            - top_k: Cantidad K de recuperados a evaluar
            - sample_limit: Filas a evaluar
        """
        if df.empty:
            raise ValueError("El DataFrame provisto para evaluación está vacío.")

        k = top_k or self.top_k
        df_eval = df.copy()
        if sample_limit and 0 < sample_limit < len(df_eval):
            logger.info(f"Limitando evaluación a las primeras {sample_limit} filas.")
            df_eval = df_eval.iloc[:sample_limit].copy().reset_index(drop=True)

        gemma_math_scores, qwen_math_scores = [], []
        gemma_suff_base, qwen_suff_base = [], []
        gemma_mrr_list, qwen_mrr_list = [], []

        gemma_llm_scores, qwen_llm_scores = [], []
        gemma_llm_reasons, qwen_llm_reasons = [], []

        gemma_hybrid_scores, qwen_hybrid_scores = [], []
        winners = []

        run_math = eval_mode in ("math", "hybrid")
        run_llm = eval_mode in ("llm", "hybrid")

        logger.info(f"Iniciando evaluación (Modo: '{eval_mode}') para {len(df_eval)} preguntas...")

        for idx, row in df_eval.iterrows():
            question_text = str(row.get("question", "")).strip()
            expected = self.extract_expected_ids(row)
            gemma_retrieved = self.parse_retrieved_list(row.get("gemma", "[]"))
            qwen_retrieved = self.parse_retrieved_list(row.get("qwen", "[]"))

            total_expected = len(expected)

            # 1. Métricas Matemáticas
            if run_math:
                if total_expected == 0:
                    g_math, q_math = 0.0, 0.0
                    g_suff, q_suff = 0.0, 0.0
                    g_mrr, q_mrr = 0.0, 0.0
                else:
                    expected_set = set(expected)
                    g_hits = len(expected_set.intersection(set(gemma_retrieved[:k])))
                    q_hits = len(expected_set.intersection(set(qwen_retrieved[:k])))

                    g_math = round(g_hits / total_expected, 4)
                    q_math = round(q_hits / total_expected, 4)

                    g_suff = 1.0 if g_hits > 0 else 0.0
                    q_suff = 1.0 if q_hits > 0 else 0.0

                    g_mrr = self.compute_mrr(expected, gemma_retrieved, top_k=k)
                    q_mrr = self.compute_mrr(expected, qwen_retrieved, top_k=k)
            else:
                g_math, q_math = 0.0, 0.0
                g_suff, q_suff = 0.0, 0.0
                g_mrr, q_mrr = 0.0, 0.0

            # 2. Evaluación Cualitativa LLM Judge
            if run_llm:
                gemma_context = self.extract_abstracts_for_row(row, gemma_retrieved[:k])
                qwen_context = self.extract_abstracts_for_row(row, qwen_retrieved[:k])

                g_llm_res = self.evaluate_llm_context_sufficiency(question_text, gemma_context)
                q_llm_res = self.evaluate_llm_context_sufficiency(question_text, qwen_context)

                g_llm_score = g_llm_res["sufficiency_score"]
                q_llm_score = q_llm_res["sufficiency_score"]
                g_llm_reason = g_llm_res["reasoning"]
                q_llm_reason = q_llm_res["reasoning"]
            else:
                g_llm_score, q_llm_score = 0.0, 0.0
                g_llm_reason, q_llm_reason = "", ""

            # 3. Score Híbrido
            if eval_mode == "math":
                g_final = g_math
                q_final = q_math
            elif eval_mode == "llm":
                g_final = g_llm_score
                q_final = q_llm_score
            else:  # hybrid
                g_final = round(0.5 * g_math + 0.5 * g_llm_score, 4)
                q_final = round(0.5 * q_math + 0.5 * q_llm_score, 4)

            gemma_math_scores.append(g_math)
            qwen_math_scores.append(q_math)
            gemma_suff_base.append(g_suff)
            qwen_suff_base.append(q_suff)
            gemma_mrr_list.append(g_mrr)
            qwen_mrr_list.append(q_mrr)

            gemma_llm_scores.append(g_llm_score)
            qwen_llm_scores.append(q_llm_score)
            gemma_llm_reasons.append(g_llm_reason)
            qwen_llm_reasons.append(q_llm_reason)

            gemma_hybrid_scores.append(g_final)
            qwen_hybrid_scores.append(q_final)

            if g_final > q_final:
                winners.append("Gemma")
            elif q_final > g_final:
                winners.append("Qwen")
            else:
                winners.append("Empate")

        detailed_df = df_eval.copy()

        if run_math:
            detailed_df["gemma_recall"] = gemma_math_scores
            detailed_df["qwen_recall"] = qwen_math_scores
            detailed_df["gemma_hit_rate"] = gemma_suff_base
            detailed_df["qwen_hit_rate"] = qwen_suff_base
            detailed_df["gemma_mrr"] = gemma_mrr_list
            detailed_df["qwen_mrr"] = qwen_mrr_list

        if run_llm:
            detailed_df["gemma_llm_score"] = gemma_llm_scores
            detailed_df["qwen_llm_score"] = qwen_llm_scores
            detailed_df["gemma_llm_reasoning"] = gemma_llm_reasons
            detailed_df["qwen_llm_reasoning"] = qwen_llm_reasons

        detailed_df["gemma_score"] = gemma_hybrid_scores
        detailed_df["qwen_score"] = qwen_hybrid_scores
        detailed_df["retrieval_winner"] = winners

        total_questions = len(detailed_df)

        def make_summary_row(
            model_name: str,
            final_scores: pd.Series,
            recalls: pd.Series,
            hits: pd.Series,
            mrrs: pd.Series,
            llm_scores: pd.Series,
        ) -> dict[str, Any]:
            row_dict: dict[str, Any] = {"Modelo": model_name}

            if eval_mode in ("math", "hybrid"):
                row_dict["Mean Recall@20"] = round(float(recalls.mean()), 4) if total_questions > 0 else 0.0
                row_dict["Hit Rate (% Base)"] = f"{round(float(hits.mean()) * 100, 2)}%"
                row_dict["Mean MRR@20"] = round(float(mrrs.mean()), 4) if total_questions > 0 else 0.0

            if eval_mode in ("llm", "hybrid"):
                row_dict["LLM Judge Sufficiency"] = round(float(llm_scores.mean()), 4) if total_questions > 0 else 0.0

            row_dict["Score Híbrido Final"] = round(float(final_scores.mean()), 4) if total_questions > 0 else 0.0
            row_dict["Preguntas Evaluadas"] = total_questions
            return row_dict

        summary_data = [
            make_summary_row(
                "Gemma",
                pd.Series(gemma_hybrid_scores),
                pd.Series(gemma_math_scores),
                pd.Series(gemma_suff_base),
                pd.Series(gemma_mrr_list),
                pd.Series(gemma_llm_scores),
            ),
            make_summary_row(
                "Qwen",
                pd.Series(qwen_hybrid_scores),
                pd.Series(qwen_math_scores),
                pd.Series(qwen_suff_base),
                pd.Series(qwen_mrr_list),
                pd.Series(qwen_llm_scores),
            ),
        ]

        summary_df = pd.DataFrame(summary_data)
        return detailed_df, summary_df

    def evaluate_retrieval_sufficiency(
        self,
        df: pd.DataFrame,
        top_k: int = 20,
        sample_limit: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluación matemática de suficiencia por omisión."""
        return self.evaluate_hybrid_sufficiency(df=df, eval_mode="math", top_k=top_k, sample_limit=sample_limit)

    def evaluate_dataframe_cv(
        self,
        df: pd.DataFrame,
        sample_limit: int | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Alias para mantener compatibilidad hacia atrás."""
        return self.evaluate_hybrid_sufficiency(df=df, eval_mode="hybrid", top_k=self.top_k, sample_limit=sample_limit)

    @staticmethod
    def format_cv_summary_table(summary_df: pd.DataFrame) -> str:
        """Formatea el DataFrame de resumen como una tabla legible en texto/consola."""
        if summary_df.empty:
            return "Sin resultados."
        return summary_df.to_string(index=False)
