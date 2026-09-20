import argparse
import logging
from pathlib import Path
import sys

from src.analytics.retrieval_evaluator import RetrievalEvaluator
from src.core.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Script de evaluación de métricas de retrieval (MRR, MAP@20, Mean RECALL@20) para Gemma y Qwen."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=config.get("paths.questions_csv", "data/3_gold/questions_v1_1000.csv"),
        help="Ruta al CSV con preguntas y documentos recuperados (por defecto: data/3_gold/questions_v1_1000.csv).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=config.get("evaluation.top_k", 20),
        help="Valor de K para top-K evaluation (por defecto: 20).",
    )
    parser.add_argument(
        "--output_summary_csv",
        type=str,
        default=config.get("evaluation.metrics_summary_csv", "data/3_gold/metrics_summary.csv"),
        help="Ruta para guardar el CSV resumen de métricas comparativas.",
    )
    parser.add_argument(
        "--output_detailed_csv",
        type=str,
        default=config.get("evaluation.evaluated_questions_csv", "data/3_gold/questions_v1_1000_evaluated.csv"),
        help="Ruta para guardar el CSV detallado con métricas por pregunta.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_summary_path = Path(args.output_summary_csv)
    output_detailed_path = Path(args.output_detailed_csv)

    logger.info(f"Iniciando evaluación de métricas de Retrieval | Dataset: '{csv_path}' | k={args.top_k}")

    try:
        evaluator = RetrievalEvaluator()
        summary_df, detailed_df = evaluator.run_evaluation(csv_path=csv_path, k=args.top_k)

        # Imprimir tabla comparativa en consola
        table_str = evaluator.format_results_table(summary_df)
        print("\n" + "=" * 70)
        print(f" TABLA COMPARATIVA DE MÉTRICAS DE RETRIEVAL (Top-{args.top_k})")
        print("=" * 70)
        print(table_str)
        print("=" * 70 + "\n")

        # Guardar archivos CSV
        output_summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_summary_path, index=False, encoding="utf-8-sig")
        logger.info(f"CSV de resumen de resultados guardado en: {output_summary_path}")

        output_detailed_path.parent.mkdir(parents=True, exist_ok=True)
        detailed_df.to_csv(output_detailed_path, index=False, encoding="utf-8-sig")
        logger.info(f"CSV de resultados detallados guardado en: {output_detailed_path}")

    except Exception as e:
        logger.error(f"Error al ejecutar la evaluación: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
