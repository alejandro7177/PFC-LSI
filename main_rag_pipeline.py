import argparse
import logging
import sys
from src.core.config import config
from src.core.factory import RAGPipelineFactory
from src.strategies.llm import Gemma12BStrategy, MockLLMStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Sistema RAG de Recuperación y Generación con Gemma 4 12B (Tuberías y Filtros)."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="¿Cuáles son los modelos principales de dinámica atmosférica?",
        help="Consulta de usuario para procesar por la tubería RAG.",
    )
    parser.add_argument(
        "--strategy",
        choices=["dense", "keyword", "mock"],
        default="mock",
        help="Estrategia de recuperación a utilizar (por defecto: mock).",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="Palabra clave opcional para delimitar el espacio vectorial en la estrategia keyword.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=config.get("rag_system.retrieval.top_k", 5),
        help="Número de documentos a recuperar.",
    )
    parser.add_argument(
        "--use_gemma_real",
        action="store_true",
        help="Si se especifica, carga los pesos reales de google/gemma-4-12B-it (requiere GPU y ~24GB VRAM).",
    )

    args = parser.parse_args()

    logger.info(f"Iniciando Sistema RAG | Estrategia: {args.strategy} | Query: '{args.query}'")

    if args.use_gemma_real:
        llm_strat = Gemma12BStrategy()
    else:
        llm_strat = MockLLMStrategy(model_name="google/gemma-4-12B-it (Simulación)")

    if args.strategy == "mock":
        pipeline = RAGPipelineFactory.create_mock_rag_pipeline()
    elif args.strategy == "dense":
        pipeline = RAGPipelineFactory.create_dense_rag_pipeline(llm_strategy=llm_strat, top_k=args.top_k)
    elif args.strategy == "keyword":
        pipeline = RAGPipelineFactory.create_keyword_rag_pipeline(
            llm_strategy=llm_strat, keyword=args.keyword, top_k=args.top_k
        )
    else:
        logger.error(f"Estrategia desconocida: {args.strategy}")
        sys.exit(1)

    initial_stream = (item for item in [args.query])
    result_stream = pipeline.run(initial_stream)

    for idx, response in enumerate(result_stream, start=1):
        print("\n" + "=" * 80)
        print(f" RESULTADO RAG #{idx}")
        print("=" * 80)
        print(f"Consulta: {response.query}")
        print(f"Modelo Base: {response.model_name}")
        print(f"Documentos Fuentes ({len(response.source_documents)}):")
        for d_idx, doc in enumerate(response.source_documents, start=1):
            print(f"  [{d_idx}] Doc ID: {doc.doc_id} | Score: {doc.score:.4f}")
            print(f"      Resumen: {doc.abstract[:120]}...")
        print("-" * 80)
        print("RESPUESTA GENERADA:")
        print(response.answer)
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
