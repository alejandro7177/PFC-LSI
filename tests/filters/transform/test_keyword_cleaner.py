from pathlib import Path
import polars as pl
import pytest
from src.filters.transform.keyword_cleaner import KeywordFilter


class TestKeywordCleaner:
    def test_init_with_json_list(self, temp_valid_keywords_json: Path):
        """Verifica que KeywordFilter cargue correctamente keywords desde un JSON con estructura de lista."""
        kf = KeywordFilter(valid_keywords_path=temp_valid_keywords_json)
        assert "python" in kf.valid_keywords
        assert "machine learning" in kf.valid_keywords
        assert "rag" in kf.valid_keywords
        assert "spacy" in kf.valid_keywords

    def test_init_with_json_dataframe_dict(self, temp_valid_keywords_dict_json: Path):
        """Verifica que KeywordFilter cargue keywords cuando el JSON tiene estructura de DataFrame dict."""
        kf = KeywordFilter(
            valid_keywords_path=temp_valid_keywords_dict_json,
            col_target="kw_v1",
        )
        assert "python" in kf.valid_keywords
        assert "machine learning" in kf.valid_keywords
        assert "rag" in kf.valid_keywords

    def test_init_unsupported_file_format(self, tmp_path: Path):
        """Verifica que se lance ValueError al proporcionar un formato de archivo no soportado."""
        invalid_file = tmp_path / "keywords.txt"
        invalid_file.write_text("python\nrag", encoding="utf-8")

        with pytest.raises(ValueError, match="Formato no soportado para keywords válidas"):
            KeywordFilter(valid_keywords_path=invalid_file)

    def test_process_filtering(self, sample_polars_df: pl.DataFrame, temp_valid_keywords_json: Path):
        """Verifica que process() filtre únicamente las keywords válidas y limpie espacios en blanco."""
        kf = KeywordFilter(
            valid_keywords_path=temp_valid_keywords_json,
            col_target="kw_v1",
            output_col="kw_v1_df30",
        )
        res_df = kf.process(sample_polars_df)

        assert "kw_v1_df30" in res_df.columns
        res_list = res_df["kw_v1_df30"].to_list()

        # Fila 1: "python, machine learning, rag" -> todas válidas
        assert sorted(res_list[0]) == sorted(["python", "machine learning", "rag"])

        # Fila 2: "  spacy , NLP  , python " -> espaciado removido
        assert "spacy" in res_list[1]
        assert "python" in res_list[1]

        # Fila 3: "c++, python-3.9, test@data!" -> caracteres especiales conservados si están en valid_keywords
        assert "c++" in res_list[2]
        assert "python-3.9" in res_list[2]

        # Fila 4: Cuestión vacía ""
        assert res_list[3] == []

        # Fila 5: None / Null (Polars produce None para str.split sobre valor nulo)
        assert res_list[4] is None

        # Fila 6: Keywords desconocidas -> lista vacía
        assert res_list[5] == []

    def test_process_custom_columns(self, temp_valid_keywords_json: Path):
        """Verifica el funcionamiento de process() usando nombres de columnas personalizados."""
        df = pl.DataFrame({"input_kw": ["python, unknown", "rag, c++"]})
        kf = KeywordFilter(
            valid_keywords_path=temp_valid_keywords_json,
            col_target="input_kw",
            output_col="filtered_kw",
        )
        res_df = kf.process(df)
        assert "filtered_kw" in res_df.columns
        assert res_df["filtered_kw"].to_list() == [["python"], ["rag", "c++"]]

    def test_process_empty_dataframe(self, empty_polars_df: pl.DataFrame, temp_valid_keywords_json: Path):
        """Verifica el comportamiento de process() con un DataFrame vacío."""
        kf = KeywordFilter(
            valid_keywords_path=temp_valid_keywords_json,
            col_target="kw_v1",
            output_col="kw_v1_df30",
        )
        res_df = kf.process(empty_polars_df)
        assert res_df.height == 0
        assert "kw_v1_df30" in res_df.columns

    def test_get_statistics(self, sample_polars_df: pl.DataFrame, temp_valid_keywords_json: Path):
        """Verifica el cálculo de estadísticas antes y después del filtrado."""
        kf = KeywordFilter(
            valid_keywords_path=temp_valid_keywords_json,
            col_target="kw_v1",
            output_col="kw_v1_df30",
        )
        processed_df = kf.process(sample_polars_df)

        stats = KeywordFilter.get_statistics(
            processed_df,
            col_before="kw_v1",
            col_after="kw_v1_df30",
        )

        assert isinstance(stats, dict)
        assert "keywords_antes" in stats
        assert "keywords_despues" in stats
        assert "documentos_sin_keywords" in stats
        assert stats["keywords_antes"] >= stats["keywords_despues"]
        assert stats["documentos_sin_keywords"] == 2  # Filas con lista de longitud 0 exactamente (excluyendo nulos)

    def test_get_statistics_empty_df(self, empty_polars_df: pl.DataFrame):
        """Verifica get_statistics() sobre un DataFrame sin filas."""
        empty_with_output = empty_polars_df.with_columns(
            pl.Series(name="kw_v1_df30", values=[], dtype=pl.List(pl.Utf8))
        )
        stats = KeywordFilter.get_statistics(
            empty_with_output,
            col_before="kw_v1",
            col_after="kw_v1_df30",
        )
        assert stats["keywords_antes"] == 0
        assert stats["keywords_despues"] == 0
        assert stats["documentos_sin_keywords"] == 0
