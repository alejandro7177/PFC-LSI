from unittest.mock import MagicMock
import polars as pl
import pytest
from src.filters.transform.keyword_normalizer import KeywordNormalizerFilter


class TestKeywordNormalizerFilterMocked:
    @pytest.fixture
    def mocked_normalizer(self, mocker):
        """Crea una instancia de KeywordNormalizerFilter con SpaCy y SpellChecker moqueados para ejecuciones ultrarrápidas."""
        mock_nlp = MagicMock()

        def fake_spacy_call(text):
            mock_doc = MagicMock()
            mock_token = MagicMock()
            # Mapeo simple sintético para simular lematización
            lemma_map = {
                "running": "run",
                "cities": "city",
                "python": "python",
                "pythn": "python",
                "machines": "machine",
            }
            mock_token.lemma_ = lemma_map.get(text, text)
            mock_doc.__len__.return_value = 1
            mock_doc.__getitem__.return_value = mock_token
            return mock_doc

        mock_nlp.side_effect = fake_spacy_call
        mocker.patch("spacy.load", return_value=mock_nlp)

        mock_spell = MagicMock()
        mock_spell.correction.side_effect = lambda word: "python" if word == "pythn" else word
        mocker.patch("src.filters.transform.keyword_normalizer.SpellChecker", return_value=mock_spell)

        normalizer = KeywordNormalizerFilter(spacy_model="en_core_web_sm", use_spellcheck=True)
        return normalizer

    def test_normalize_single_keyword_with_mocks(self, mocked_normalizer: KeywordNormalizerFilter):
        """Verifica la normalización individual de términos usando mocks."""
        # Corrección ortográfica + lematización
        assert mocked_normalizer.normalize_single_keyword("pythn") == "python"
        # Lematización estándar
        assert mocked_normalizer.normalize_single_keyword("running") == "run"
        assert mocked_normalizer.normalize_single_keyword("cities") == "city"

    def test_normalize_single_keyword_empty_returns_empty(self, mocked_normalizer: KeywordNormalizerFilter):
        """Verifica que cadenas vacías o con solo espacios retornen cadena vacía."""
        assert mocked_normalizer.normalize_single_keyword("") == ""
        assert mocked_normalizer.normalize_single_keyword("   ") == ""

    def test_normalize_vocabulary(self, mocked_normalizer: KeywordNormalizerFilter):
        """Verifica que normalize_vocabulary deduplique y ordene alfabéticamente el resultado."""
        input_keywords = ["running", "cities", "running", "", "  ", "pythn"]
        result = mocked_normalizer.normalize_vocabulary(input_keywords)
        assert result == ["city", "python", "run"]

    def test_process_string_column(self, mocked_normalizer: KeywordNormalizerFilter, sample_polars_df: pl.DataFrame):
        """Verifica la transformación sobre una columna de tipo cadena (pl.Utf8)."""
        res_df = mocked_normalizer.process(
            sample_polars_df,
            input_col="keywords_qwen",
            output_col="keywords_normalized",
        )

        assert "keywords_normalized" in res_df.columns
        norm_list = res_df["keywords_normalized"].to_list()
        assert isinstance(norm_list[0], list)
        assert "machine" in norm_list[0] or "python" in norm_list[0]

    def test_process_list_column(self, mocked_normalizer: KeywordNormalizerFilter, sample_polars_df: pl.DataFrame):
        """Verifica la transformación sobre una columna que ya contiene listas (pl.List(pl.Utf8))."""
        res_df = mocked_normalizer.process(
            sample_polars_df,
            input_col="kw_list",
            output_col="keywords_normalized",
        )

        assert "keywords_normalized" in res_df.columns
        norm_list = res_df["keywords_normalized"].to_list()
        assert isinstance(norm_list, list)

    def test_process_missing_column_raises_value_error(self, mocked_normalizer: KeywordNormalizerFilter, sample_polars_df: pl.DataFrame):
        """Verifica que se lance ValueError cuando la columna de entrada no exista en el DataFrame."""
        with pytest.raises(ValueError, match="La columna 'non_existent_col' no existe"):
            mocked_normalizer.process(sample_polars_df, input_col="non_existent_col")

    def test_process_empty_dataframe(self, mocked_normalizer: KeywordNormalizerFilter, empty_polars_df: pl.DataFrame):
        """Verifica el procesamiento sobre un DataFrame vacío."""
        res_df = mocked_normalizer.process(
            empty_polars_df,
            input_col="keywords_qwen",
            output_col="keywords_normalized",
        )
        assert res_df.height == 0
        assert "keywords_normalized" in res_df.columns


class TestKeywordNormalizerIntegration:
    def test_real_spacy_and_spellcheck_integration(self):
        """Prueba de integración real con SpaCy y SpellChecker sin mocks si el modelo está presente."""
        try:
            normalizer = KeywordNormalizerFilter(spacy_model="en_core_web_sm", use_spellcheck=False)
            res = normalizer.normalize_single_keyword("running")
            assert res == "run"
        except OSError:
            pytest.skip("Modelo SpaCy 'en_core_web_sm' no instalado en el entorno actual.")
