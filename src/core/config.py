from pathlib import Path
from typing import Any
import yaml


class NestedConfig:
    """Clase auxiliar para permitir acceso encadenado por atributos, p. ej. config.paths.default_input_csv."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            val = self._data[name]
            if isinstance(val, dict):
                return NestedConfig(val)
            return val
        raise AttributeError(f"'NestedConfig' no tiene el atributo '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        if isinstance(val, dict):
            return NestedConfig(val)
        return val

    def __getitem__(self, item: str) -> Any:
        val = self._data[item]
        if isinstance(val, dict):
            return NestedConfig(val)
        return val

class ConfigLoader:
    """Carga y gestiona la configuración del proyecto desde YAML de forma limpia y robusta."""

    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            # Buscar relativo a la raíz del proyecto (2 niveles arriba de src/core/config.py)
            root_dir = Path(__file__).resolve().parents[2]
            resolved_path = root_dir / "config" / "config.yaml"
            if resolved_path.exists():
                self.config_path = resolved_path
            else:
                self.config_path = Path("config/config.yaml")
        else:
            self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de configuración en: {self.config_path}")

        self._config: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get(self, key_path: str, default: Any = None) -> Any:
        """Permite acceder a claves anidadas usando notación de punto, e.g. 'keyword_normalizer.spacy_model'."""
        keys = key_path.split(".")
        val: Any = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def __getitem__(self, key_path: str) -> Any:
        val = self.get(key_path)
        if val is None and key_path not in self._config:
            raise KeyError(f"Clave no encontrada en configuración: '{key_path}'")
        return val

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"'ConfigLoader' no tiene el atributo '{name}'")
        if name in self._config:
            val = self._config[name]
            if isinstance(val, dict):
                return NestedConfig(val)
            return val
        raise AttributeError(f"'ConfigLoader' no tiene el atributo '{name}'")


# Instancia global para importar directo desde src.core.config import config
config = ConfigLoader()