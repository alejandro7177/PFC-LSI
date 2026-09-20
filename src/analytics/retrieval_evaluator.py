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


class RetrievalEvaluator:
    """Evaluador de modelos de recuperación de información (Gemma vs Qwen)."""

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

