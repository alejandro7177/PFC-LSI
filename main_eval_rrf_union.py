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
        description="Script de evaluación de Ensamble RRF y Límite Empírico Máximo (Unión Qwen y Gemma)."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=config.get("paths.questions_csv", "data/3_gold/questions_v1_1000.csv"),
        help="Ruta al CSV con preguntas y recuperados (por defecto: data/3_gold/questions_v1_1000.csv).",
    )
    parser.add_argument(
        "--k_rrf",
        type=int,
        default=60,
        help="Constante k para el cálculo de RRF (por defecto: 60).",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=config.get("evaluation.top_k", 20),
        help="Número de documentos Top-N a considerar (por defecto: 20).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=config.get("evaluation.questions_v2_rrf_union_csv", "data/3_gold/questions_v2_rrf_union.csv"),
        help="Ruta para guardar el CSV con los resultados de RRF y Unión.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_path = Path(args.output_csv)

    logger.info(
        f"Iniciando Evaluación RRF y Unión | Dataset: '{csv_path}' | k_rrf={args.k_rrf} | top_n={args.top_n}"
    )

    try:
        evaluator = RetrievalEvaluator()
        summary_df, detailed_df = evaluator.run_rrf_and_union_evaluation(
            csv_path=csv_path,
            k_rrf=args.k_rrf,
            top_n=args.top_n,
        )

        table_str = evaluator.format_results_table(summary_df)
        print("\n" + "=" * 80)
        print(" TABLA COMPARATIVA GLOBAL (Qwen vs Gemma vs RRF vs Unión Máxima)")
        print("=" * 80)
        print(table_str)
        print("=" * 80 + "\n")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        detailed_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Resultados de RRF y Unión guardados exitosamente en: {output_path}")
        summary_df.to_csv("data/3_gold/rrf_union_summary.csv", index=False, encoding="utf-8-sig")
        logger.info("Tabla de resumen de resultados guardada exitosamente en: data/3_gold/rrf_union_summary.csv")

    except Exception as e:
        logger.error(f"Error durante la ejecución de RRF y Unión: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
