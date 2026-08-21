import json
from pathlib import Path
import pytest
import polars as pl
import yaml
from src.core.config import ConfigLoader


@pytest.fixture
def sample_polars_df() -> pl.DataFrame:
    """Proporciona un DataFrame de Polars sintético con casos borde (nulos, vacíos, caracteres especiales)."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "kw_v1": [
                "python, machine learning, rag",
                "  spacy , NLP  , python ",
                "c++, python-3.9, test@data!",
                "",
                None,
                "unknown_kw1, unknown_kw2",
            ],
            "keywords_qwen": [
                "python, machine learning",
                "spacy, nlp",
                "c++, python-3.9",
                "",
                None,
                "unknown_kw1",
            ],
            "kw_v1_df30": [
                ["python", "machine learning", "rag"],
                ["spacy", "nlp"],
                ["c++"],
                [],
                [],
                [],
            ],
            "kw_list": [
                ["python", "machine learning"],
                ["spacy", "nlp"],
                ["c++", "python-3.9"],
                [],
                None,
                ["unknown_kw1"],
            ],
        }
    )


@pytest.fixture
def empty_polars_df() -> pl.DataFrame:
    """Proporciona un DataFrame de Polars completamente vacío con el esquema esperado."""
    return pl.DataFrame(
        schema={
            "id": pl.Int64,
            "kw_v1": pl.Utf8,
            "keywords_qwen": pl.Utf8,
            "kw_v1_df30": pl.List(pl.Utf8),
            "kw_list": pl.List(pl.Utf8),
        }
    )


@pytest.fixture
def temp_valid_keywords_json(tmp_path: Path) -> Path:
    """Crea un archivo JSON temporal con estructura de registros/DataFrame compatible con polars.read_json."""
    valid_keywords = [
        {"kw_v1": "python"},
        {"kw_v1": "machine learning"},
        {"kw_v1": "rag"},
        {"kw_v1": "spacy"},
        {"kw_v1": "nlp"},
        {"kw_v1": "c++"},
        {"kw_v1": "python-3.9"},
        {"kw_v1": "test@data!"},
    ]
    file_path = tmp_path / "valid_keywords_list.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(valid_keywords, f)
    return file_path


@pytest.fixture
def temp_valid_keywords_dict_json(tmp_path: Path) -> Path:
    """Crea un archivo JSON temporal con otra columna ('keyword') para probar el fallback a to_series()."""
    valid_keywords_dict = [
        {"keyword": "python"},
        {"keyword": "machine learning"},
        {"keyword": "rag"},
        {"keyword": "spacy"},
        {"keyword": "nlp"},
    ]
    file_path = tmp_path / "valid_keywords_dict.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(valid_keywords_dict, f)
    return file_path


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Crea un archivo YAML de configuración sintético e aislado para pruebas."""
    config_dict = {
        "project": {
            "name": "test-project",
            "version": "0.1.0",
        },
        "paths": {
            "valid_keywords_json": str(tmp_path / "valid_keywords_list.json"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "keyword_cleaner": {
            "col_target": "kw_v1",
            "output_col": "kw_v1_df30",
            "delimiter": ",",
        },
        "keyword_normalizer": {
            "spacy_model": "en_core_web_sm",
            "disable_components": ["parser", "ner"],
            "use_spellcheck": True,
            "language": "en",
            "input_col": "keywords_qwen",
            "output_col": "keywords_normalized",
        },
        "visualization": {
            "dpi": 100,
            "top_n_bar_chart": 5,
            "wordcloud": {
                "width": 400,
                "height": 200,
                "background_color": "white",
            },
            "histogram": {
                "bins": 10,
            },
        },
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)
    return config_path


@pytest.fixture
def mock_config_loader(temp_config_file: Path) -> ConfigLoader:
    """Instancia un ConfigLoader aislado usando la configuración de prueba."""
    return ConfigLoader(config_path=temp_config_file)
