import json
from unittest.mock import MagicMock, PropertyMock, patch
import pytest
from src.filters.transform.keyword_extractor import KeywordExtractorFilter


class TestKeywordExtractorFilter:
    @pytest.fixture
    def mock_kw_model(self):
        with patch.object(KeywordExtractorFilter, "kw_model", new_callable=PropertyMock) as mock_prop:
            mock_model = MagicMock()
            mock_model.extract_keywords.return_value = [("quantum", 0.9), ("physics", 0.8)]
            mock_prop.return_value = mock_model
            yield mock_model

    def test_process_with_vocab(self, tmp_path, mock_kw_model):
        vocab_file = tmp_path / "vocab.json"
        vocab_file.write_text(json.dumps(["quantum", "physics"]), encoding="utf-8")

        extractor = KeywordExtractorFilter(vocab_path=str(vocab_file))
        stream = [{"abstract": "Quantum physics is fascinating."}]
        result = list(extractor.process(stream))

        assert len(result) == 1
        assert result[0]["keywords"] == "quantum, physics"
        mock_kw_model.extract_keywords.assert_called_once()
        assert mock_kw_model.extract_keywords.call_args.kwargs.get("vectorizer") is not None

    def test_process_without_vocab_file_not_found(self, tmp_path, mock_kw_model):
        non_existent_vocab = tmp_path / "non_existent.json"

        extractor = KeywordExtractorFilter(vocab_path=str(non_existent_vocab))
        stream = [{"abstract": "Quantum physics is fascinating."}]
        result = list(extractor.process(stream))

        assert len(result) == 1
        assert result[0]["keywords"] == "quantum, physics"
        mock_kw_model.extract_keywords.assert_called_once()
        assert mock_kw_model.extract_keywords.call_args.kwargs.get("vectorizer") is None

    def test_process_empty_abstract(self, mock_kw_model):
        extractor = KeywordExtractorFilter(vocab_path="non_existent.json")
        stream = [{"abstract": ""}]
        result = list(extractor.process(stream))

        assert len(result) == 1
        assert result[0]["keywords"] == ""
        mock_kw_model.extract_keywords.assert_not_called()
