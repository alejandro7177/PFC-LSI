import pandas as pd
from src.core.base import Pipeline
from src.core.config import config
from src.filters.extract.arxiv_extractor import ArxivExtractor
from src.filters.transform.keyword_extractor import KeywordExtractorFilter


def run_download_pipeline():
    max_results = config.get("arxiv_extractor.max_results", 100)
    extractor = ArxivExtractor(max_results=max_results)

    # Creamos un stream a partir de la API de arXiv
    raw_stream = extractor.stream_papers()

    # Ensamblamos la tubería
    pipeline = Pipeline()
    pipeline.add_filter(KeywordExtractorFilter())

    # Ejecutamos y guardamos resultados procesados
    processed_data = list(pipeline.run(raw_stream))
    df = pd.DataFrame(processed_data)
    out_file = config.get("paths.papers_processed_csv", "data/1_raw/papers_processed.csv")
    df.to_csv(out_file, index=False)
    print(f"Pipeline completado. Resultados guardados en: {out_file}")


if __name__ == "__main__":
    run_download_pipeline()