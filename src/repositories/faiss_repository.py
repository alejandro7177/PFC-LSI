import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    doc_id: str
    score: float
    abstract: str
    keywords: str


class FAISSRepository:
    def __init__(
        self,
        model_name: str | None = None,
        use_gpu: bool | None = None,
        normalize: bool | None = None
    ):
        self.model_name = model_name or config.get("faiss_repository.model_name", "Qwen/Qwen3-Embedding-8B")
        self.use_gpu = use_gpu if use_gpu is not None else config.get("faiss_repository.use_gpu", True)
        self.normalize = normalize if normalize is not None else config.get("faiss_repository.normalize", True)

        device = "cuda:1" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        logger.info(f"Loading embedding model {self.model_name} on {device}...")
        self.embedding_model = SentenceTransformer(self.model_name, device=device)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: list[dict[str, Any]] = []
        self._doc_id_to_index: dict[str, int] = {}
        self.keyword_mapping: dict[str, list[int]] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        ).astype(np.float32)

        if self.normalize:
            faiss.normalize_L2(embeddings)
        return embeddings

    def add_documents(self, documents: list[dict[str, Any]]):
        if not documents:
            return

        texts = [doc["abstract"] for doc in documents]
        embeddings = self.encode(texts)

        self.index.add(embeddings)
        start_idx = len(self.metadata)

        output_col = config.get("keyword_cleaner.output_col", "kw_v1_df30")
        for i, doc in enumerate(documents):
            meta = {
                "doc_id": str(doc["doc_id"]),
                "abstract": doc["abstract"],
                "keywords": doc.get("keywords", doc.get(output_col, ""))
            }
            self.metadata.append(meta)
            self._doc_id_to_index[meta["doc_id"]] = start_idx + i

    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        k = k if k is not None else config.get("faiss_repository.top_k", 5)
        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty.")

        query_emb = self.encode([query])
        scores, indices = self.index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append(
                SearchResult(
                    doc_id=meta["doc_id"],
                    score=float(score),
                    abstract=meta["abstract"],
                    keywords=meta["keywords"]
                )
            )
        return results

    def save(self, directory: str | Path | None = None):
        directory = Path(directory or config.get("faiss_repository.vector_store_dir", "vectorStores/Qwen-8B"))
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "faiss.index"))
        with open(directory / "metadata.json", "w", encoding="utf8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)

        conf = {
            "model": self.model_name,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "documents": self.index.ntotal
        }
        with open(directory / "config.json", "w", encoding="utf8") as f:
            json.dump(conf, f, indent=4)

    def load(self, directory: str | Path | None = None):
        directory = Path(directory or config.get("faiss_repository.vector_store_dir", "vectorStores/Qwen-8B"))
        self.index = faiss.read_index(str(directory / "faiss.index"))
        with open(directory / "metadata.json", encoding="utf8") as f:
            self.metadata = json.load(f)
        self._doc_id_to_index = {str(meta["doc_id"]): i for i, meta in enumerate(self.metadata)}

    def load_keyword_mapping(self, mapping_path: str | Path | None = None) -> dict[str, list[int]]:
        """Carga el archivo JSON que mapea palabras clave a sus respectivos doc_ids."""
        mapping_path = Path(mapping_path or config.get("faiss_repository.mapping_path", "data/2_processed/keywords_mapping.json"))
        if not mapping_path.exists():
            raise FileNotFoundError(f"Archivo de mapeo no encontrado: {mapping_path}")

        logger.info("Cargando mapeo de keywords desde %s...", mapping_path)
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.keyword_mapping.clear()

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "doc_id" in item:
                    kw_key = next((k for k in item.keys() if k not in ("doc_id", "df")), None)
                    if kw_key and item[kw_key]:
                        kw_name = str(item[kw_key]).strip().lower()
                        self.keyword_mapping[kw_name] = item["doc_id"]
        elif isinstance(data, dict):
            for kw, doc_ids in data.items():
                if isinstance(doc_ids, list):
                    self.keyword_mapping[str(kw).strip().lower()] = doc_ids

        return self.keyword_mapping

    def search_by_doc_ids(
        self,
        query: str,
        doc_ids: list[int | str],
        k: int | None = None
    ) -> list[SearchResult]:
        """Realiza búsqueda vectorial únicamente sobre el subconjunto delimitado por doc_ids."""
        k = k if k is not None else config.get("faiss_repository.top_k", 5)
        if self.index.ntotal == 0:
            raise RuntimeError("El índice FAISS está vacío.")

        if not self._doc_id_to_index:
            self._doc_id_to_index = {str(meta["doc_id"]): i for i, meta in enumerate(self.metadata)}

        # Filtrar solo los índices FAISS que pertenecen a los doc_ids candidatos
        valid_indices = [
            self._doc_id_to_index[str(d_id)]
            for d_id in doc_ids
            if str(d_id) in self._doc_id_to_index
        ]

        if not valid_indices:
            logger.warning("No se encontraron coincidencias de doc_ids en el índice.")
            return []

        # Reconstrucción de vectores del subconjunto (Operación en memoria)
        sub_embeddings = np.vstack([self.index.reconstruct(i) for i in valid_indices]).astype(np.float32)
        if self.normalize:
            faiss.normalize_L2(sub_embeddings)

        query_embedding = self.encode([query])
        scores = np.dot(sub_embeddings, query_embedding.T).flatten()

        top_k_positions = np.argsort(-scores)[:k]

        results = []
        for pos in top_k_positions:
            orig_idx = valid_indices[pos]
            meta = self.metadata[orig_idx]
            results.append(
                SearchResult(
                    doc_id=meta["doc_id"],
                    score=float(scores[pos]),
                    abstract=meta["abstract"],
                    keywords=meta["keywords"]
                )
            )
        return results

    def search_by_keyword(
        self,
        keyword: str,
        query: str | None = None,
        mapping_path: str | Path | None = None,
        k: int | None = None
    ) -> list[SearchResult]:
        """
        Delimita la búsqueda del espacio vectorial a una sola keyword usando el mapa JSON.
        """
        k = k if k is not None else config.get("faiss_repository.top_k", 5)
        clean_kw = str(keyword).strip().lower()
        if not clean_kw:
            raise ValueError("La keyword no puede estar vacía.")

        if mapping_path is not None:
            self.load_keyword_mapping(mapping_path)
        elif not self.keyword_mapping:
            default_mapping = Path(config.get("faiss_repository.mapping_path", "data/2_processed/keywords_mapping.json"))
            if default_mapping.exists():
                self.load_keyword_mapping(default_mapping)
            else:
                raise FileNotFoundError("Carga un mapeo con load_keyword_mapping() primero.")

        if clean_kw not in self.keyword_mapping:
            raise KeyError(f"La keyword '{keyword}' no existe en el mapeo cargado.")

        candidate_doc_ids = self.keyword_mapping[clean_kw]
        search_query = query if query is not None else keyword

        return self.search_by_doc_ids(query=search_query, doc_ids=candidate_doc_ids, k=k)