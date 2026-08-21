from abc import ABC, abstractmethod
from typing import Generator, Any

class Filter(ABC):
    @abstractmethod
    def process(self, stream: Generator[Any, None, None]) -> Generator[Any, None, None]:
        """Recibe un stream de datos, aplica la transformación y emite un stream."""
        pass

class Pipeline:
    def __init__(self):
        self._filters: list[Filter] = []

    def add_filter(self, filter_step: Filter) -> 'Pipeline':
        self._filters.append(filter_step)
        return self

    def run(self, initial_stream: Generator[Any, None, None]) -> Generator[Any, None, None]:
        stream = initial_stream
        for f in self._filters:
            stream = f.process(stream)
        return stream