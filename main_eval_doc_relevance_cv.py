import argparse
import logging
from pathlib import Path
import sys
import pandas as pd

from src.analytics.document_relevance_evaluator import DocumentRelevanceEvaluator
from src.core.config import config
from src.strategies.llm import Gemma12BStrategy, MockLLMStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación Híbrida de Suficiencia y Rendimiento de Recuperación de Documentos (Gemma vs Qwen)."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=config.get("paths.questions_csv", "data/3_gold/questions_v1_1000.csv"),
        help="Ruta al CSV con preguntas y documentos recuperados (por defecto: data/3_gold/questions_v1_1000.csv).",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["math", "llm", "hybrid"],
        default="hybrid",
        help="Modo de evaluación: 'math' (Ground Truth), 'llm' (LLM Judge) o 'hybrid' (Ambas combinadas). Por defecto: hybrid.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=config.get("evaluation.top_k", 20),
        help="Cantidad K de documentos recuperados a considerar (por defecto: 20).",
    )
    parser.add_argument(
        "--use_gemma_real",
        action="store_true",
        help="Si se especifica, utiliza la estrategia real Gemma 4 12B en GPU para el LLM Judge. Por defecto usa MockLLMStrategy.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Límite opcional de preguntas a evaluar para pruebas rápidas.",
    )
    parser.add_argument(
        "--output_summary_csv",
        type=str,
        default=config.get("doc_relevance_cv.output_summary_csv", "data/3_gold/doc_relevance_cv_summary.csv"),
        help="Ruta para guardar el CSV resumen de métricas de suficiencia.",
    )
    parser.add_argument(
        "--output_detailed_csv",
        type=str,
        default=config.get("doc_relevance_cv.output_detailed_csv", "data/3_gold/doc_relevance_cv_detailed.csv"),
        help="Ruta para guardar el CSV detallado con scores por pregunta.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_summary_path = Path(args.output_summary_csv)
    output_detailed_path = Path(args.output_detailed_csv)

    if not csv_path.exists():
        logger.error(f"El archivo dataset '{csv_path}' no existe.")
        sys.exit(1)

    logger.info(f"Cargando dataset desde '{csv_path}'...")
    df = pd.read_csv(csv_path)

    if args.use_gemma_real:
        logger.info("Inicializando Gemma 4 12B Real en GPU para LLM Judge...")
        llm_strategy = Gemma12BStrategy()
    else:
        logger.info("Inicializando MockLLMStrategy para evaluación híbrida de prueba...")
        llm_strategy = MockLLMStrategy(model_name="Gemma-4-12B-SufficiencyJudge (Mock)")

    evaluator = DocumentRelevanceEvaluator(
        llm_strategy=llm_strategy,
        top_k=args.top_k,
    )

    logger.info(
        f"Iniciando Evaluación Híbrida de Recuperación (Modo: '{args.eval_mode}' | Top-{args.top_k} docs para Gemma y Qwen)..."
    )
    detailed_df, summary_df = evaluator.evaluate_hybrid_sufficiency(
        df=df,
        eval_mode=args.eval_mode,
        top_k=args.top_k,
        sample_limit=args.sample_limit,
    )

    table_str = evaluator.format_cv_summary_table(summary_df)
    print("\n" + "=" * 85)
    print(f" RESUMEN DE EVALUACIÓN HÍBRIDA DE RECUPERACIÓN (GEMMA vs QWEN | Modo: {args.eval_mode.upper()})")
    print("=" * 85)
    print(table_str)
    print("=" * 85 + "\n")

    # Guardar resultados
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_summary_path, index=False, encoding="utf-8-sig")
    logger.info(f"Resumen guardado en: {output_summary_path}")

    output_detailed_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_df.to_csv(output_detailed_path, index=False, encoding="utf-8-sig")
    logger.info(f"Scores detallados por pregunta guardados en: {output_detailed_path}")


if __name__ == "__main__":
    main()
