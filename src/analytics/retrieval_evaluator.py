import json
import logging
from pathlib import Path
import pandas as pd

from src.core.config import config

logger = logging.getLogger(__name__)


class RetrievalMetricsCalculator:
    """Calculador de métricas individuales y grupales de Information Retrieval (IR)."""

    @staticmethod
    def reciprocal_rank(expected_ids: list[str], retrieved_ids: list[str], k: int = 20) -> float:
        """
        Calcula el Reciprocal Rank (RR) para una consulta.
        Retorna 1/r donde r es la posición de la primera coincidencia en el top-k.
        Si no hay coincidencia, retorna 0.0.
        """
        if not expected_ids:
            return 0.0

        expected_set = set(expected_ids)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in expected_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def average_precision(expected_ids: list[str], retrieved_ids: list[str], k: int = 20) -> float:
        """
        Calcula el Average Precision (AP@k) para una consulta.
        AP@k = (1 / |R|) * sum_{j=1}^k (Precision@j * rel(j))
        donde |R| es la cantidad total de documentos relevantes esperados.
        """
        if not expected_ids:
            return 0.0

        expected_set = set(expected_ids)
        hits = 0
        sum_precisions = 0.0

        for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in expected_set:
                hits += 1
                precision_at_j = hits / rank
                sum_precisions += precision_at_j

        return sum_precisions / len(expected_ids)

    @staticmethod
    def recall_at_k(expected_ids: list[str], retrieved_ids: list[str], k: int = 20) -> float:
        """
        Calcula el Recall@k para una consulta.
        Recall@k = |R cap L_{1..k}| / |R|
        """
        if not expected_ids:
            return 0.0

        expected_set = set(expected_ids)
        retrieved_set = set(retrieved_ids[:k])
        matches = len(expected_set.intersection(retrieved_set))
        return matches / len(expected_ids)

    @staticmethod
    def compute_rrf(qwen_list: list[str], gemma_list: list[str], k: int = 60, top_n: int = 20) -> list[str]:
        """
        Calcula la fusión RRF (Reciprocal Rank Fusion) para dos listas de recuperados.
        Score(d) = 1/(k + r_qwen(d)) + 1/(k + r_gemma(d))
        Retorna los top_n documentos con mayor score.
        """
        scores: dict[str, float] = {}

        for rank, doc_id in enumerate(qwen_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        for rank, doc_id in enumerate(gemma_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

        # Ordenar documentos por score descendente
        sorted_docs = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        return sorted_docs[:top_n]

    @staticmethod
    def compute_union_limit(qwen_list: list[str], gemma_list: list[str]) -> list[str]:
        """
        Calcula la unión sin duplicados de las listas de recuperados de Qwen y Gemma.
        Preserva el orden de aparición.
        """
        seen = set()
        union_docs = []
        for doc_id in qwen_list + gemma_list:
            if doc_id not in seen:
                seen.add(doc_id)
                union_docs.append(doc_id)
        return union_docs


class RetrievalEvaluator:
    """Evaluador de modelos de recuperación de información (Gemma vs Qwen, RRF y Unión)."""

    def __init__(self, doc_id_cols: list[str] | None = None):
        self.doc_id_cols = doc_id_cols or [f"doc_id_{i}" for i in range(1, 6)]
        self.calculator = RetrievalMetricsCalculator()

    @staticmethod
    def _parse_retrieved_list(val: str | list) -> list[str]:
        """Parsea la cadena JSON o lista de documentos recuperados a lista de strings."""
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

    def _extract_expected_ids(self, row: pd.Series) -> list[str]:
        """Extrae los IDs esperados relevantes de una fila del CSV."""
        expected_ids = []
        for col in self.doc_id_cols:
            if col in row and pd.notna(row[col]):
                val = row[col]
                try:
                    expected_ids.append(str(int(val)))
                except (ValueError, TypeError):
                    expected_ids.append(str(val).strip())
        return expected_ids

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        model_name: str,
        retrieved_col: str,
        k: int = 20
    ) -> tuple[dict[str, float], pd.DataFrame]:
        """
        Evalúa un modelo en el DataFrame dado para una columna de recuperados.
        Retorna (métricas promedio, DataFrame con columnas de métricas individuales agregadas).
        """
        mrr_list = []
        map_list = []
        recall_list = []

        for _, row in df.iterrows():
            expected = self._extract_expected_ids(row)
            retrieved = self._parse_retrieved_list(row.get(retrieved_col, "[]"))

            rr = self.calculator.reciprocal_rank(expected, retrieved, k=k)
            ap = self.calculator.average_precision(expected, retrieved, k=k)
            rec = self.calculator.recall_at_k(expected, retrieved, k=k)

            mrr_list.append(round(rr, 4))
            map_list.append(round(ap, 4))
            recall_list.append(round(rec, 4))

        eval_df = df.copy()
        eval_df[f"{model_name}_mrr"] = mrr_list
        eval_df[f"{model_name}_map{k}"] = map_list
        eval_df[f"{model_name}_recall{k}"] = recall_list

        summary_metrics = {
            "Model": model_name.capitalize(),
            f"MRR@{k}": round(pd.Series(mrr_list).mean(), 4),
            f"MAP@{k}": round(pd.Series(map_list).mean(), 4),
            f"Mean RECALL@{k}": round(pd.Series(recall_list).mean(), 4),
        }

        return summary_metrics, eval_df

    def run_evaluation(
        self,
        csv_path: str | Path,
        k: int = 20
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carga el CSV, realiza la evaluación de Gemma y Qwen, y devuelve
        (DataFrame resumen de comparación, DataFrame completo evaluado).
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo CSV en: {csv_path}")

        logger.info(f"Cargando dataset para evaluación desde: {csv_path}")
        df = pd.read_csv(csv_path)

        gemma_summary, df_evaluated = self.evaluate_dataframe(df, "gemma", "gemma", k=k)
        qwen_summary, df_evaluated = self.evaluate_dataframe(df_evaluated, "qwen", "qwen", k=k)

        summary_df = pd.DataFrame([gemma_summary, qwen_summary])
        return summary_df, df_evaluated

    def run_rrf_and_union_evaluation(
        self,
        csv_path: str | Path,
        k_rrf: int = 60,
        top_n: int = 20
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calcula RRF y Unión empírica fila por fila conservando todas las columnas originales.
        Devuelve (summary_df, detailed_df con rrf_docs, rrf_res, rrf_mrr, union_docs, union_res).
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo CSV en: {csv_path}")

        logger.info(f"Cargando dataset para RRF y Unión desde: {csv_path}")
        df = pd.read_csv(csv_path)

        qwen_mrr_list = []
        qwen_recall_list = []
        gemma_mrr_list = []
        gemma_recall_list = []

        rrf_docs_list = []
        rrf_res_list = []
        rrf_mrr_list = []

        union_docs_list = []
        union_res_list = []

        for _, row in df.iterrows():
            expected = self._extract_expected_ids(row)
            qwen_list = self._parse_retrieved_list(row.get("qwen", "[]"))
            gemma_list = self._parse_retrieved_list(row.get("gemma", "[]"))

            # Métricas individuales de modelos
            q_mrr = self.calculator.reciprocal_rank(expected, qwen_list, k=top_n)
            q_rec = self.calculator.recall_at_k(expected, qwen_list, k=top_n)
            g_mrr = self.calculator.reciprocal_rank(expected, gemma_list, k=top_n)
            g_rec = self.calculator.recall_at_k(expected, gemma_list, k=top_n)

            qwen_mrr_list.append(round(q_mrr, 4))
            qwen_recall_list.append(round(q_rec, 4))
            gemma_mrr_list.append(round(g_mrr, 4))
            gemma_recall_list.append(round(g_rec, 4))

            # RRF (Ensamble)
            rrf_docs = self.calculator.compute_rrf(qwen_list, gemma_list, k=k_rrf, top_n=top_n)
            rrf_rec = self.calculator.recall_at_k(expected, rrf_docs, k=top_n)
            rrf_mrr = self.calculator.reciprocal_rank(expected, rrf_docs, k=top_n)

            rrf_docs_list.append(json.dumps(rrf_docs))
            rrf_res_list.append(round(rrf_rec, 4))
            rrf_mrr_list.append(round(rrf_mrr, 4))

            # Unión Empírica Máxima
            union_docs = self.calculator.compute_union_limit(qwen_list, gemma_list)
            union_rec = self.calculator.recall_at_k(expected, union_docs, k=len(union_docs))

            union_docs_list.append(json.dumps(union_docs))
            union_res_list.append(round(union_rec, 4))

        detailed_df = df.copy()
        detailed_df["rrf_docs"] = rrf_docs_list
        detailed_df["rrf_res"] = rrf_res_list
        detailed_df["rrf_mrr"] = rrf_mrr_list
        detailed_df["union_docs"] = union_docs_list
        detailed_df["union_res"] = union_res_list

        total_queries = len(df)

        def get_summary_row(name: str, recall_series: pd.Series, mrr_series: pd.Series | None):
            zero_failures = int((recall_series == 0.0).sum())
            fail_pct = (zero_failures / total_queries) * 100 if total_queries > 0 else 0.0
            mrr_val = f"{mrr_series.mean():.4f}" if mrr_series is not None else "N/A"
            return {
                "Enfoque": name,
                "Mean Recall@20": f"{recall_series.mean():.4f}",
                "Mean MRR": mrr_val,
                "Fallos Totales (Recall=0.0)": zero_failures,
                "% Fallos": f"{fail_pct:.2f}%",
            }

        summary_rows = [
            get_summary_row("Qwen", pd.Series(qwen_recall_list), pd.Series(qwen_mrr_list)),
            get_summary_row("Gemma", pd.Series(gemma_recall_list), pd.Series(gemma_mrr_list)),
            get_summary_row("RRF (Ensamble)", pd.Series(rrf_res_list), pd.Series(rrf_mrr_list)),
            get_summary_row("Unión Máxima (Qwen ∪ Gemma)", pd.Series(union_res_list), None),
        ]

        summary_df = pd.DataFrame(summary_rows)
        return summary_df, detailed_df

    @staticmethod
    def format_results_table(summary_df: pd.DataFrame) -> str:
        """Formatea el DataFrame de métricas como una tabla Markdown sin dependencias externas."""
        headers = list(summary_df.columns)
        rows = summary_df.values.tolist()

        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(val)))

        header_line = "| " + " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
        separator_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
        row_lines = [
            "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
            for row in rows
        ]
        return "\n".join([header_line, separator_line] + row_lines)
