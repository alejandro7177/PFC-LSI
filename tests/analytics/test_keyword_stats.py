from pathlib import Path
import pytest
from src.analytics.keywords_stats import KeywordVisualizer


@pytest.fixture
def sample_keyword_count() -> dict[str, int]:
    """Proporciona un diccionario sintético de frecuencias de keywords para pruebas de gráficos."""
    return {
        "python": 150,
        "machine learning": 120,
        "deep learning": 90,
        "rag": 75,
        "spacy": 50,
        "nlp": 45,
        "polars": 30,
        "faiss": 20,
        "transformers": 15,
        "pytest": 10,
    }


class TestKeywordVisualizer:
    def test_ensure_dir_creates_directory(self, tmp_path: Path):
        """Verifica que _ensure_dir cree carpetas y subcarpetas si no existen."""
        target_dir = tmp_path / "nested" / "reports" / "figures"
        assert not target_dir.exists()

        created_dir = KeywordVisualizer._ensure_dir(target_dir)
        assert created_dir.exists()
        assert created_dir.is_dir()

    def test_histogram_generation(self, sample_keyword_count: dict[str, int], tmp_path: Path):
        """Verifica que histogram() genere el archivo histogram.png."""
        KeywordVisualizer.histogram(sample_keyword_count, tmp_path, bins=5)
        output_file = tmp_path / "histogram.png"
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_top_keywords_generation(self, sample_keyword_count: dict[str, int], tmp_path: Path):
        """Verifica que top_keywords() genere el archivo top_<n>.png especificado."""
        top_n = 5
        KeywordVisualizer.top_keywords(sample_keyword_count, tmp_path, top_n=top_n)
        output_file = tmp_path / f"top_{top_n}.png"
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_boxplot_generation(self, sample_keyword_count: dict[str, int], tmp_path: Path):
        """Verifica que boxplot() genere el archivo boxplot.png."""
        KeywordVisualizer.boxplot(sample_keyword_count, tmp_path)
        output_file = tmp_path / "boxplot.png"
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_zipf_plot_generation(self, sample_keyword_count: dict[str, int], tmp_path: Path):
        """Verifica que zipf_plot() genere el archivo zipf.png."""
        KeywordVisualizer.zipf_plot(sample_keyword_count, tmp_path)
        output_file = tmp_path / "zipf.png"
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_wordcloud_generation(self, sample_keyword_count: dict[str, int], tmp_path: Path):
        """Verifica que wordcloud() genere el archivo wordcloud.png."""
        KeywordVisualizer.wordcloud(sample_keyword_count, tmp_path)
        output_file = tmp_path / "wordcloud.png"
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_generate_all_plots(self, sample_keyword_count: dict[str, int], tmp_path: Path):
        """Verifica que generate_all_plots() ejecute y genere el paquete completo de 5 gráficos."""
        output_dir = tmp_path / "all_plots"
        KeywordVisualizer.generate_all_plots(sample_keyword_count, output_dir, top_n=5)

        expected_files = [
            "histogram.png",
            "top_5.png",
            "boxplot.png",
            "zipf.png",
            "wordcloud.png",
        ]
        for file_name in expected_files:
            file_path = output_dir / file_name
            assert file_path.exists(), f"Falta el archivo generado: {file_name}"
            assert file_path.stat().st_size > 0, f"El archivo {file_name} está vacío"

    def test_edge_case_single_keyword(self, tmp_path: Path):
        """Verifica la generación de gráficos con un único elemento en el diccionario."""
        single_kw = {"python": 100}
        KeywordVisualizer.generate_all_plots(single_kw, tmp_path, top_n=1)
        assert (tmp_path / "top_1.png").exists()

    def test_edge_case_empty_dictionary(self, tmp_path: Path):
        """Verifica el comportamiento de los gráficos cuando se proporciona un diccionario vacío."""
        empty_kw: dict[str, int] = {}
        # histogram, boxplot y top_keywords deben ejecutarse sin errores
        KeywordVisualizer.histogram(empty_kw, tmp_path)
        KeywordVisualizer.top_keywords(empty_kw, tmp_path, top_n=5)
        KeywordVisualizer.boxplot(empty_kw, tmp_path)

        assert (tmp_path / "histogram.png").exists()
        assert (tmp_path / "top_5.png").exists()
        assert (tmp_path / "boxplot.png").exists()

        # zipf_plot lanza ValueError al intentar aplicar log en datos sin valores positivos
        with pytest.raises(ValueError, match="log-scaled"):
            KeywordVisualizer.zipf_plot(empty_kw, tmp_path)
