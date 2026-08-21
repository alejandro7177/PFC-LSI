# main_generate_questions.py
import argparse
from pathlib import Path
import pandas as pd

from src.core.config import config
from src.filters.transform.question_generator import (
    QuestionGeneratorFilter,
    extract_keyword_groups,
)


def main():
    parser = argparse.ArgumentParser(description="Generador ELT de preguntas sintéticas.")
    parser.add_argument("--csv_path", type=str, default=config.get("paths.dataset_csv", "data/dataset.csv"))
    parser.add_argument("--output_csv_path", type=str, default=config.get("paths.questions_csv", "data/questions_v1_1000.csv"))
    parser.add_argument("--prompt_v1_path", type=str, default=config.get("question_generator.prompt_path", "prompts/question_generation_v1.md"))
    parser.add_argument("--kw_column", type=str, default=config.get("question_generator.kw_column", "kw_v1_df30"))
    parser.add_argument("--num_questions", type=int, default=config.get("question_generator.num_questions", 1000))
    parser.add_argument("--batch_size", type=int, default=config.get("question_generator.batch_size", 16))
    parser.add_argument("--model_name", type=str, default=config.get("question_generator.model_name", "google/gemma-4-12B-it"))
    parser.add_argument("--cuda_gpu", type=str, default=str(config.get("question_generator.cuda_gpu", "1")))
    parser.add_argument("--random_state", type=int, default=config.get("question_generator.random_state", 42))

    args = parser.parse_args()

    # 1. Stream de extracción
    stream = extract_keyword_groups(
        csv_path=args.csv_path,
        kw_column=args.kw_column,
        num_questions=args.num_questions,
        random_state=args.random_state,
    )

    # 2. Filtro de generación
    filter_gen = QuestionGeneratorFilter(
        model_name=args.model_name,
        prompt_path=args.prompt_v1_path,
        cuda_gpu=args.cuda_gpu,
        batch_size=args.batch_size,
    )

    # 3. Procesar stream
    processed_stream = filter_gen.process(stream)

    # 4. Exportar a CSV
    results = []
    for item in processed_stream:
        row_dict = {
            "question": item["question"],
            "keyword": item["keyword"],
        }
        for pos, (doc_id, abstract) in enumerate(zip(item["doc_ids"], item["abstracts"]), 1):
            row_dict[f"doc_id_{pos}"] = doc_id
            row_dict[f"abstract_{pos}"] = abstract
        results.append(row_dict)

    out_path = Path(args.output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Preguntas guardadas con éxito en: {out_path}")


if __name__ == "__main__":
    main()