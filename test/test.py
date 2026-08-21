import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from faissVectorStore import FaissVectorStore
from src.core.config import config


def evaluate_qwen_retrieval(
    csv_path: str | Path | None = None,
    vector_store_dir: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    question_column: str | None = None,
    keyword_column: str | Path | None = None,
    kw_mapping_path: str | Path | None = None,
    n_samples: int | None = None,
    k: int | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Toma una muestra aleatoria de `n_samples` filas de `dataset.csv`, evalúa las
    preguntas de `question_column` en la base de vectores FAISS (Qwen) e identifica
    si recupera el documento correspondiente (`doc_id`).
    """
    csv_path = Path(csv_path or config.get("paths.dataset_csv", "data/dataset.csv"))
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo CSV en: {csv_path}")

    vector_store_dir = Path(vector_store_dir or config.get("faiss_repository.gemma_vector_store_dir", "vectorStores/gemma4"))
    if not vector_store_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio del vector store en: {vector_store_dir}")

    output_csv_path = Path(output_csv_path or config.get("evaluation.eval_retrieval_output_csv", "data/eval_retrieval_gemma4.csv"))
    question_column = question_column or config.get("question_cleaner.output_col", "question_gemma4_12b_clean")
    kw_mapping_path = kw_mapping_path or config.get("faiss_repository.mapping_path", "data/2_processed/keywords_mapping.json")
    n_samples = n_samples or config.get("evaluation.n_samples", 2000)
    k = k or config.get("evaluation.top_k", 20)
    random_state = random_state if random_state is not None else config.get("evaluation.random_state", 17)

    print(f"Cargando dataset desde {csv_path}...")
    df = pd.read_csv(csv_path)

    if question_column not in df.columns:
        raise ValueError(
            f"La columna '{question_column}' no existe en el CSV. Columnas disponibles: {list(df.columns)}"
        )
    if "doc_id" not in df.columns:
        raise ValueError("La columna 'doc_id' no existe en el CSV.")

    # Filtrar filas donde la pregunta no sea nula ni vacía
    valid_df = df[df[question_column].notna() & (df[question_column].str.strip() != "")].copy()

    if len(valid_df) < n_samples:
        print(f"Advertencia: El número de filas válidas ({len(valid_df)}) es menor que n_samples ({n_samples}). Usando todas.")
        sampled_df = valid_df
    else:
        sampled_df = valid_df.sample(n=n_samples, random_state=random_state)

    print(f"Cargando Vector Store desde {vector_store_dir}...")
    store = FaissVectorStore(
        csv_path=csv_path,
        model_name=config.get("faiss_repository.gemma_model_name", "google/embeddinggemma-300m")
    )
    store.load(vector_store_dir)

    if keyword_column and kw_mapping_path:
        store.load_keyword_mapping(kw_mapping_path)

    results_data = []

    print(f"Evaluando {len(sampled_df)} preguntas en el Vector Store (k={k})...")
    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Evaluando preguntas"):
        expected_doc_id = row["doc_id"]
        question = row[question_column]

        # Búsqueda en el vector store (general o delimitada por keyword)
        if keyword_column and pd.notna(row.get(keyword_column)):
            raw_kw = str(row[keyword_column]).split(",")[0].strip()
            try:
                search_results = store.search_by_keyword(
                    keyword=raw_kw,
                    query=question,
                    mapping_path=kw_mapping_path,
                    k=k
                )
            except Exception:
                search_results = store.search(question, k=k)
        else:
            search_results = store.search(question, k=k)

        retrieved_doc_ids = [res.doc_id for res in search_results]

        # Verificar si el documento esperado está en los recuperados
        found = str(expected_doc_id) in [str(doc_id) for doc_id in retrieved_doc_ids]

        results_data.append(
            {
                "doc_id": expected_doc_id,
                "question": question,
                "retrieved_doc_ids": json.dumps(retrieved_doc_ids),
                "found": found,
            }
        )
    out_df = pd.DataFrame(results_data)

    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    stadisct_eval(out_df, k)
    print(f"Resultados guardados en: {out_path}")

    return out_df


def stadisct_eval(out_df: pd.DataFrame, k: int):
    total = len(out_df)
    found_count = out_df["found"].sum()
    accuracy = (found_count / total) * 100 if total > 0 else 0.0

    print(f"\n--- Resumen de Evaluación ---")
    print(f"Total de preguntas evaluadas: {total}")
    print(f"Documentos esperados encontrados (top-{k}): {found_count}/{total} ({accuracy:.2f}%)")


def evaluate_questions_v1_1000(
    csv_path: str | Path | None = None,
    model_type: str = "gemma",
    k: int | None = None,
) -> pd.DataFrame:
    """
    Evalúa las preguntas de `questions_v1_1000.csv` utilizando el Vector Store especificado
    ('gemma' o 'qwen') recuperando k documentos por pregunta.
    """
    csv_path = Path(csv_path or config.get("paths.questions_csv", "data/questions_v1_1000.csv"))
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo CSV en: {csv_path}")

    k = k or config.get("evaluation.top_k", 20)
    model_type = model_type.lower()
    if model_type not in ("gemma", "qwen"):
        raise ValueError("model_type debe ser 'gemma' o 'qwen'")

    if model_type == "gemma":
        vector_store_dir = Path(config.get("faiss_repository.gemma_vector_store_dir", "vectorStores/gemma4"))
        model_name = config.get("faiss_repository.gemma_model_name", "google/embeddinggemma-300m")
    else:
        vector_store_dir = Path(config.get("faiss_repository.vector_store_dir", "vectorStores/Qwen-8B"))
        model_name = config.get("faiss_repository.model_name", "Qwen/Qwen3-Embedding-8B")

    if not vector_store_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio del vector store en: {vector_store_dir}")

    print(f"\n==================================================")
    print(f" Evaluando: {csv_path}")
    print(f" Modelo: {model_type.upper()} ({model_name})")
    print(f" Vector Store Dir: {vector_store_dir} | k={k}")
    print(f"==================================================")

    df = pd.read_csv(csv_path)

    doc_id_cols = [f"doc_id_{i}" for i in range(1, 6)]
    for col in doc_id_cols:
        if col not in df.columns:
            raise ValueError(f"La columna esperada '{col}' no existe en el CSV.")

    if "question" not in df.columns:
        raise ValueError("La columna 'question' no existe en el CSV.")

    print(f"Cargando Vector Store ({model_type.upper()})...")
    dataset_csv = config.get("paths.dataset_csv", "data/dataset.csv")
    store = FaissVectorStore(
        csv_path=dataset_csv,
        model_name=model_name,
        use_gpu=True,
    )
    store.load(vector_store_dir)

    retrieved_list = []
    scores_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluando preguntas ({model_type.upper()})"):
        expected_ids = []
        for col in doc_id_cols:
            val = row[col]
            if pd.notna(val):
                try:
                    expected_ids.append(str(int(val)))
                except (ValueError, TypeError):
                    expected_ids.append(str(val).strip())

        question = str(row["question"])
        raw_kw = str(row["keyword"])

        search_results = store.search_by_keyword(raw_kw, question, k=k)
        retrieved_ids = [str(res.doc_id) for res in search_results]

        matches = len(set(expected_ids).intersection(set(retrieved_ids)))
        total_expected = len(expected_ids) if len(expected_ids) > 0 else 5
        score = matches / total_expected

        retrieved_list.append(json.dumps(retrieved_ids))
        scores_list.append(round(score, 4))

    col_retrieved = model_type
    col_res = f"{model_type}_res"

    df[col_retrieved] = retrieved_list
    df[col_res] = scores_list

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV cargado y guardado en: {csv_path}")

    avg_score = df[col_res].mean() * 100
    print(f"\n--- Resumen de Evaluación [{model_type.upper()}] ---")
    print(f"Total de preguntas evaluadas: {len(df)}")
    print(f"Porcentaje promedio de acierto por pregunta (Recall@{k}): {avg_score:.2f}%")
    print("Distribución de aciertos (de 5 docs esperados):")
    res_counts = df[col_res].value_counts().sort_index()
    for res_val, count in res_counts.items():
        pct = (count / len(df)) * 100
        n_docs = int(round(res_val * 5))
        print(f"  - {n_docs}/5 aciertos ({res_val:.1f}): {count} preguntas ({pct:.1f}%)")

    return df


if __name__ == "__main__":
    k_eval = config.get("evaluation.top_k", 20)
    questions_csv = config.get("paths.questions_csv", "data/questions_v1_1000.csv")
    evaluate_questions_v1_1000(csv_path=questions_csv, model_type="gemma", k=k_eval)
    evaluate_questions_v1_1000(csv_path=questions_csv, model_type="qwen", k=k_eval)
