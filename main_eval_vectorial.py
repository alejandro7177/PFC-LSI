import argparse
import logging
import sys
from pathlib import Path

from src.analytics.vector_search_evaluator import VectorSearchEvaluator
from src.core.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta búsqueda vectorial top-k con Gemma y Qwen para cada pregunta del CSV de entrada "
            "y guarda los doc_ids recuperados como columnas 'gemma' y 'qwen' en el CSV de salida."
        )
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/2_processed/question_finals(res).csv",
        help="Ruta al CSV de entrada con preguntas (por defecto: data/2_processed/question_finals(res).csv).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=config.get("evaluation.vector_search_output_csv", "data/3_gold/question_finals_evaluated.csv"),
        help="Ruta del CSV de salida con los resultados de búsqueda vectorial agregados.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Número de documentos a recuperar por búsqueda (por defecto: 40).",
    )
    parser.add_argument(
        "--qwen_store_dir",
        type=str,
        default=None,
        help="Directorio del vector store de Qwen (por defecto: config faiss_repository.vector_store_dir).",
    )
    parser.add_argument(
        "--gemma_store_dir",
        type=str,
        default=None,
        help="Directorio del vector store de Gemma (por defecto: config faiss_repository.gemma_vector_store_dir).",
    )
    parser.add_argument(
        "--question_col",
        type=str,
        default="question",
        help="Nombre de la columna con el texto de la pregunta en el CSV (por defecto: 'question').",
    )
    parser.add_argument(
        "--overwrite_cols",
        action="store_true",
        help="Si se especifica, sobreescribe las columnas 'gemma' y 'qwen' si ya existen en el CSV.",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Si se especifica, desactiva el uso de GPU para los modelos de embeddings.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_path = Path(args.output_path)

    if not csv_path.exists():
        logger.error(f"El CSV de entrada '{csv_path}' no existe.")
        sys.exit(1)

    use_gpu = not args.no_gpu

    logger.info("=" * 70)
    logger.info(" EVALUACIÓN DE BÚSQUEDA VECTORIAL (Gemma & Qwen)")
    logger.info("=" * 70)
    logger.info(f"  CSV de entrada : {csv_path}")
    logger.info(f"  CSV de salida  : {output_path}")
    logger.info(f"  Top-K          : {args.top_k}")
    logger.info(f"  GPU habilitada : {use_gpu}")
    logger.info("=" * 70)

    try:
        evaluator = VectorSearchEvaluator(
            qwen_store_dir=args.qwen_store_dir,
            gemma_store_dir=args.gemma_store_dir,
            top_k=args.top_k,
            use_gpu=use_gpu,
        )

        df = evaluator.evaluate(
            csv_path=csv_path,
            output_path=output_path,
            top_k=args.top_k,
            question_col=args.question_col,
            overwrite_cols=args.overwrite_cols,
        )

        # Resumen rápido en consola
        total = len(df)
        qwen_ok = df["qwen"].apply(lambda x: x != "[]").sum()
        gemma_ok = df["gemma"].apply(lambda x: x != "[]").sum()

        print("\n" + "=" * 70)
        print(" RESUMEN DE EVALUACIÓN VECTORIAL")
        print("=" * 70)
        print(f"  Preguntas procesadas  : {total}")
        print(f"  Qwen — con resultados : {qwen_ok}/{total}")
        print(f"  Gemma — con resultados: {gemma_ok}/{total}")
        print(f"  CSV de salida         : {output_path}")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Error durante la evaluación vectorial: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
