# src/filters/transform/keyword_cleaner.py
from dataclasses import dataclass, field
from pathlib import Path
import polars as pl
from src.core.base import Filter
from src.core.config import config


@dataclass
class KeywordFilter(Filter):
    valid_keywords_path: str | Path | None = None
    col_target: str | None = None
    output_col: str | None = None
    valid_keywords: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        if self.valid_keywords_path is None:
            self.valid_keywords_path = config.get("paths.valid_keywords_json", "data/artifacts/kw_mapping_df30.json")
        if self.col_target is None:
            self.col_target = config.get("keyword_cleaner.col_target", "kw_v1")
        if self.output_col is None:
            self.output_col = config.get("keyword_cleaner.output_col", "kw_v1_df30")

        path = Path(self.valid_keywords_path)
        if path.suffix == ".json":
            key_valid = pl.read_json(path)
            # Soporta estructura de lista directa o columna en dataframe
            if self.col_target in key_valid.columns:
                self.valid_keywords = set(key_valid[self.col_target].to_list())
            else:
                self.valid_keywords = set(key_valid.to_series().to_list())
        else:
            raise ValueError(f"Formato no soportado para keywords válidas: {path.suffix}")

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aplica el filtrado de keywords conservando únicamente las válidas."""
        delimiter = config.get("keyword_cleaner.delimiter", ",")
        return df.with_columns(
            pl.col(self.col_target)
            .str.split(delimiter)
            .list.eval(
                pl.when(pl.element().str.strip_chars().is_in(self.valid_keywords))
                .then(pl.element().str.strip_chars())
                .otherwise(None)
            )
            .list.drop_nulls()
            .alias(self.output_col)
        )

    @staticmethod
    def get_statistics(
        df: pl.DataFrame,
        col_before: str | None = None,
        col_after: str | None = None
    ) -> dict[str, int]:
        """Calcula estadísticas antes y después del filtrado."""
        col_before = col_before or config.get("keyword_cleaner.col_target", "kw_v1")
        col_after = col_after or config.get("keyword_cleaner.output_col", "kw_v1_df30")
        delimiter = config.get("keyword_cleaner.delimiter", ",")

        antes = df[col_before].str.split(delimiter).list.len().sum()
        despues = df[col_after].list.len().sum()
        docs_sin_kw = df.filter(pl.col(col_after).list.len() == 0).height

        return {
            "keywords_antes": int(antes),
            "keywords_despues": int(despues),
            "documentos_sin_keywords": int(docs_sin_kw),
        }