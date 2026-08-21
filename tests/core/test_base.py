# tests/core/test_base.py
import pytest
from src.core.base import Filter, Pipeline


class DummyFilter(Filter):
    def process(self, stream):
        for item in stream:
            yield item * 2


def test_base_filter():
    class DirectFilter(Filter):
        def process(self, stream):
            return stream

    f = DirectFilter()
    assert f.process([1, 2]) == [1, 2]


def test_pipeline():
    p = Pipeline()
    f1 = DummyFilter()
    f2 = DummyFilter()

    ret = p.add_filter(f1).add_filter(f2)
    assert ret is p

    initial_stream = (x for x in [1, 2, 3])
    result = list(p.run(initial_stream))
    assert result == [4, 8, 12]