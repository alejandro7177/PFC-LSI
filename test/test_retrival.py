import sys
from pathlib import Path

# Fix sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import config
from src.repositories.faiss_repository import FAISSRepository


def main():
    # 1. Instanciar el repositorio usando la configuración centralizada
    model_name = config.get("faiss_repository.model_name", "Qwen/Qwen3-Embedding-8B")
    repo = FAISSRepository(model_name=model_name)

    # 2. Cargar el índice ya construido por el pipeline ELT
    vector_dir = config.get("faiss_repository.vector_store_dir", "vectorStores/Qwen-8B")
    repo.load(vector_dir)

    # 3. Cargar mapeo de keywords almacenado
    mapping_path = config.get("faiss_repository.mapping_path", "data/2_processed/keywords_mapping.json")
    repo.load_keyword_mapping(mapping_path)

    # 4. Ejecutar la consulta restringida por Keyword
    top_k = config.get("faiss_repository.top_k", 3)
    results = repo.search_by_keyword(
        keyword="vegetation",
        query="What is the safe operating space of the Amazon rainforest?",
        k=top_k
    )

    for r in results:
        print(f"Doc ID: {r.doc_id} | Score: {r.score:.4f}")
        print(f"Keywords: {r.keywords}")
        print(f"Abstract: {r.abstract[:150]}...\n")


if __name__ == "__main__":
    main()