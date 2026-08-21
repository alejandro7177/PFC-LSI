# main_preprocess_keywords.py
import argparse
import json
from datetime import datetime
from pathlib import Path
import polars as pl

from src.analytics.keywords_stats import KeywordVisualizer
from src.core.config import config
from src.filters.transform.keyword_cleaner import KeywordFilter


def main():
    parser = argparse.ArgumentParser(description="Preprocesamiento y analítica de Keywords")
    parser.add_argument("--action", choices=["filter", "visualize"], required=True)
    parser.add_argument("--input_csv", type=str, default=config.get("paths.default_input_csv", "data/1_raw/dataset.csv"))
    parser.add_argument("--valid_kw_json", type=str, default=config.get("paths.valid_keywords_json", "data/kw_mapping_df30.json"))
    parser.add_argument("--output_dir", type=str, default=config.get("paths.processed_data_dir", "data/2_processed"))
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_col = config.get("keyword_cleaner.output_col", "kw_v1_df30")

    if args.action == "filter":
        print(f"Cargando dataset desde {args.input_csv}...")
        df = pl.read_csv(args.input_csv)

        cleaner = KeywordFilter(valid_keywords_path=args.valid_kw_json)
        df_cleaned = cleaner.process(df)

        # Convertir lista a string separado por comas para exportar a CSV
        delimiter = config.get("keyword_cleaner.delimiter", ",")
        df_export = df_cleaned.with_columns(pl.col(output_col).list.join(f"{delimiter} "))

        output_file = out_dir / f"dataset_filtered_{timestamp}.csv"
        df_export.write_csv(output_file)
        print(f"Dataset filtrado guardado en: {output_file}")

        # Guardar métricas
        stats = KeywordFilter.get_statistics(df_cleaned)
        stats_file = out_dir / f"stats_keywords_{timestamp}.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        print(f"Estadísticas guardadas en: {stats_file}")

    elif args.action == "visualize":
        print(f"Generando visualizaciones desde mapa {args.valid_kw_json}...")
        kw_mapping = pl.read_json(args.valid_kw_json)

        # Asumiendo mapeo con columnas [output_col, df]
        if output_col in kw_mapping.columns:
            keyword_count = dict(zip(kw_mapping[output_col], kw_mapping["df"]))
        else:
            first_col = kw_mapping.columns[0]
            keyword_count = dict(zip(kw_mapping[first_col], kw_mapping["df"]))

        plots_dir = out_dir / f"plots_{timestamp}"
        KeywordVisualizer.generate_all_plots(keyword_count, plots_dir)
        print(f"Visualizaciones guardadas en: {plots_dir}")


if __name__ == "__main__":
    main()