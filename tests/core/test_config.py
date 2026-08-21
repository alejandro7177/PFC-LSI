from pathlib import Path
import pytest
import yaml
from src.core.config import ConfigLoader, NestedConfig


class TestConfigLoader:
    def test_load_valid_config(self, temp_config_file: Path):
        """Verifica que ConfigLoader cargue correctamente un archivo YAML existente."""
        loader = ConfigLoader(config_path=temp_config_file)
        assert loader.config_path == temp_config_file
        assert loader.get("project.name") == "test-project"

    def test_default_config_path_fallback(self):
        """Verifica el comportamiento de inicialización por defecto cuando no se pasa config_path."""
        loader = ConfigLoader()
        assert loader.config_path.exists()
        assert loader.get("project.name") == "rag-keyword-preprocessing"

    def test_non_existent_file_raises_error(self, tmp_path: Path):
        """Verifica que se lance FileNotFoundError al intentar cargar un archivo que no existe."""
        non_existent_path = tmp_path / "non_existent.yaml"
        with pytest.raises(FileNotFoundError, match="No se encontró el archivo de configuración"):
            ConfigLoader(config_path=non_existent_path)

    def test_get_nested_keys_with_dot_notation(self, mock_config_loader: ConfigLoader):
        """Verifica el acceso a claves anidadas usando notación de puntos."""
        assert mock_config_loader.get("keyword_cleaner.col_target") == "kw_v1"
        assert mock_config_loader.get("visualization.wordcloud.width") == 400
        assert mock_config_loader.get("project.version") == "0.1.0"

    def test_get_non_existent_key_returns_default(self, mock_config_loader: ConfigLoader):
        """Verifica que get() retorne el valor por defecto si la clave no existe."""
        assert mock_config_loader.get("non.existent.key") is None
        assert mock_config_loader.get("non.existent.key", "fallback_value") == "fallback_value"
        assert mock_config_loader.get("keyword_cleaner.invalid_sub_key", 123) == 123

    def test_getitem_access(self, mock_config_loader: ConfigLoader):
        """Verifica el acceso mediante corchetes (dict-like)."""
        assert mock_config_loader["keyword_cleaner.col_target"] == "kw_v1"
        assert mock_config_loader["project.name"] == "test-project"

    def test_getitem_non_existent_key_raises_key_error(self, mock_config_loader: ConfigLoader):
        """Verifica que __getitem__ lance KeyError si la clave no existe."""
        with pytest.raises(KeyError, match="Clave no encontrada en configuración"):
            _ = mock_config_loader["non_existent_key"]

    def test_attribute_access(self, mock_config_loader: ConfigLoader):
        """Verifica el acceso a la configuración vía atributos estilo objeto."""
        assert mock_config_loader.project.name == "test-project"
        assert mock_config_loader.keyword_cleaner.delimiter == ","
        assert isinstance(mock_config_loader.visualization, NestedConfig)

    def test_attribute_access_non_existent_raises_attribute_error(self, mock_config_loader: ConfigLoader):
        """Verifica que acceder a un atributo inexistente lance AttributeError."""
        with pytest.raises(AttributeError, match="'ConfigLoader' no tiene el atributo"):
            _ = mock_config_loader.invalid_section

    def test_private_attribute_access_raises_attribute_error(self, mock_config_loader: ConfigLoader):
        """Verifica que el acceso a atributos privados que no existen en la clase lance AttributeError."""
        with pytest.raises(AttributeError, match="'ConfigLoader' no tiene el atributo '_unknown_private'"):
            _ = mock_config_loader._unknown_private

    def test_empty_yaml_file(self, tmp_path: Path):
        """Verifica el comportamiento cuando el archivo YAML está completamente vacío."""
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("", encoding="utf-8")

        loader = ConfigLoader(config_path=empty_yaml)
        assert loader.get("any.key") is None
        assert loader.get("any.key", "default") == "default"
        with pytest.raises(KeyError):
            _ = loader["missing"]


class TestNestedConfig:
    def test_nested_config_getattr(self):
        """Verifica el acceso a atributos en NestedConfig."""
        nested = NestedConfig({"a": 1, "b": {"c": 2}})
        assert nested.a == 1
        assert isinstance(nested.b, NestedConfig)
        assert nested.b.c == 2

    def test_nested_config_getattr_raises_attribute_error(self):
        """Verifica que acceder a una clave inexistente en NestedConfig lance AttributeError."""
        nested = NestedConfig({"a": 1})
        with pytest.raises(AttributeError, match="'NestedConfig' no tiene el atributo 'invalid'"):
            _ = nested.invalid

    def test_nested_config_get(self):
        """Verifica el método get() de NestedConfig."""
        nested = NestedConfig({"a": 1, "b": {"c": 2}})
        assert nested.get("a") == 1
        assert isinstance(nested.get("b"), NestedConfig)
        assert nested.get("non_existent", "default") == "default"

    def test_nested_config_getitem(self):
        """Verifica __getitem__ en NestedConfig."""
        nested = NestedConfig({"a": 1, "b": {"c": 2}})
        assert nested["a"] == 1
        assert isinstance(nested["b"], NestedConfig)
        with pytest.raises(KeyError):
            _ = nested["invalid"]
