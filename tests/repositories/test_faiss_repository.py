import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.repositories.faiss_repository import FAISSRepository


@pytest.fixture
def mock_sentence_transformer():
    with patch("src.repositories.faiss_repository.SentenceTransformer") as mock_st_cls:
        mock_st_inst = MagicMock()
        mock_st_inst.get_sentence_embedding_dimension.return_value = 128
        
        def mock_encode(texts, **kwargs):
            n = len(texts) if isinstance(texts, list) else 1
            return np.zeros((n, 128), dtype=np.float32)

        mock_st_inst.encode.side_effect = mock_encode
        mock_st_cls.return_value = mock_st_inst
        yield mock_st_cls, mock_st_inst


@pytest.fixture
def faiss_repo(mock_sentence_transformer):
    repo = FAISSRepository(model_name="dummy-model", use_gpu=False, normalize=True)
    return repo


def test_faiss_repository_init(mock_sentence_transformer):
    mock_st_cls, mock_st_inst = mock_sentence_transformer
    repo = FAISSRepository(model_name="dummy-model", use_gpu=False)
    assert repo.model_name == "dummy-model"
    assert repo.dimension == 128
    mock_st_cls.assert_called_once_with("dummy-model", device="cpu")


def test_faiss_repository_encode(faiss_repo):
    with patch("src.repositories.faiss_repository.faiss.normalize_L2") as mock_norm:
        emb = faiss_repo.encode(["hello world"])
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (1, 128)
        mock_norm.assert_called_once()


def test_faiss_repository_add_documents(faiss_repo):
    docs = [
        {"doc_id": 1, "abstract": "first document", "keywords": "kw1, kw2"},
        {"doc_id": 2, "abstract": "second document", "keywords": "kw3"},
    ]
    faiss_repo.add_documents(docs)
    assert faiss_repo.index.ntotal == 2
    assert len(faiss_repo.metadata) == 2
    assert faiss_repo._doc_id_to_index["1"] == 0
    assert faiss_repo._doc_id_to_index["2"] == 1


def test_faiss_repository_add_empty_documents(faiss_repo):
    faiss_repo.add_documents([])
    assert faiss_repo.index.ntotal == 0


def test_faiss_repository_search_empty_index_raises(faiss_repo):
    with pytest.raises(RuntimeError, match="Index is empty"):
        faiss_repo.search("test query")


def test_faiss_repository_search(faiss_repo):
    docs = [
        {"doc_id": "d1", "abstract": "abs 1", "keywords": "k1"},
        {"doc_id": "d2", "abstract": "abs 2", "keywords": "k2"},
    ]
    faiss_repo.add_documents(docs)
    
    mock_index = MagicMock()
    mock_index.ntotal = 2
    mock_index.search.return_value = (
        np.array([[0.95, 0.80]], dtype=np.float32),
        np.array([[0, 1]], dtype=np.int64)
    )
    faiss_repo.index = mock_index

    results = faiss_repo.search("query", k=2)
    assert len(results) == 2
    assert results[0].doc_id == "d1"
    assert results[0].score == pytest.approx(0.95)
    assert results[1].doc_id == "d2"


def test_faiss_repository_save_and_load(faiss_repo, tmp_path):
    docs = [{"doc_id": "d1", "abstract": "abs 1", "keywords": "k1"}]
    faiss_repo.add_documents(docs)

    save_dir = tmp_path / "faiss_store"
    with patch("src.repositories.faiss_repository.faiss.write_index") as mock_write, \
         patch("src.repositories.faiss_repository.faiss.read_index") as mock_read:
        
        mock_read.return_value = MagicMock()
        faiss_repo.save(save_dir)

        assert (save_dir / "metadata.json").exists()
        assert (save_dir / "config.json").exists()
        mock_write.assert_called_once()

        faiss_repo.load(save_dir)
        mock_read.assert_called_once()
        assert faiss_repo._doc_id_to_index["d1"] == 0


def test_faiss_repository_load_keyword_mapping_list(faiss_repo, tmp_path):
    mapping_data = [
        {"doc_id": [1, 2], "python": "python", "df": 2},
        {"doc_id": [3], "nlp": "nlp", "df": 1},
    ]
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(mapping_data), encoding="utf-8")

    mapping = faiss_repo.load_keyword_mapping(path)
    assert mapping["python"] == [1, 2]
    assert mapping["nlp"] == [3]


def test_faiss_repository_load_keyword_mapping_dict(faiss_repo, tmp_path):
    mapping_data = {
        "python": [10, 20],
        "machine learning": [30]
    }
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(mapping_data), encoding="utf-8")

    mapping = faiss_repo.load_keyword_mapping(path)
    assert mapping["python"] == [10, 20]
    assert mapping["machine learning"] == [30]


def test_faiss_repository_load_keyword_mapping_not_found(faiss_repo, tmp_path):
    with pytest.raises(FileNotFoundError):
        faiss_repo.load_keyword_mapping(tmp_path / "non_existent.json")


def test_faiss_repository_search_by_doc_ids(faiss_repo):
    docs = [
        {"doc_id": "10", "abstract": "abs 10", "keywords": "k10"},
        {"doc_id": "20", "abstract": "abs 20", "keywords": "k20"},
    ]
    faiss_repo.add_documents(docs)

    mock_index = MagicMock()
    mock_index.ntotal = 2
    mock_index.reconstruct.side_effect = lambda i: np.ones(128, dtype=np.float32) * (i + 1)
    faiss_repo.index = mock_index

    results = faiss_repo.search_by_doc_ids(query="test", doc_ids=["10", "20"], k=2)
    assert len(results) == 2


def test_faiss_repository_search_by_doc_ids_empty_or_invalid(faiss_repo):
    with pytest.raises(RuntimeError, match="El índice FAISS está vacío"):
        faiss_repo.search_by_doc_ids(query="test", doc_ids=["1"])

    docs = [{"doc_id": "10", "abstract": "abs 10", "keywords": "k10"}]
    faiss_repo.add_documents(docs)
    results = faiss_repo.search_by_doc_ids(query="test", doc_ids=["999"])
    assert results == []


def test_faiss_repository_search_by_keyword(faiss_repo, tmp_path):
    docs = [
        {"doc_id": "10", "abstract": "abs 10", "keywords": "python"},
        {"doc_id": "20", "abstract": "abs 20", "keywords": "python"},
    ]
    faiss_repo.add_documents(docs)
    
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps({"python": ["10", "20"]}), encoding="utf-8")

    mock_index = MagicMock()
    mock_index.ntotal = 2
    mock_index.reconstruct.side_effect = lambda i: np.ones(128, dtype=np.float32)
    faiss_repo.index = mock_index

    results = faiss_repo.search_by_keyword(keyword="python", query="python code", mapping_path=mapping_path, k=2)
    assert len(results) == 2


def test_faiss_repository_search_with_unmatched_idx(faiss_repo):
    docs = [{"doc_id": "d1", "abstract": "abs 1", "keywords": "k1"}]
    faiss_repo.add_documents(docs)
    
    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_index.search.return_value = (
        np.array([[0.95, 0.0]], dtype=np.float32),
        np.array([[0, -1]], dtype=np.int64)
    )
    faiss_repo.index = mock_index

    results = faiss_repo.search("query", k=2)
    assert len(results) == 1
    assert results[0].doc_id == "d1"


def test_faiss_repository_search_by_doc_ids_rebuilds_doc_id_map(faiss_repo):
    faiss_repo.metadata = [{"doc_id": "10", "abstract": "abs 10", "keywords": "k10"}]
    faiss_repo._doc_id_to_index = {}

    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_index.reconstruct.return_value = np.ones(128, dtype=np.float32)
    faiss_repo.index = mock_index

    results = faiss_repo.search_by_doc_ids(query="test", doc_ids=["10"], k=1)
    assert len(results) == 1
    assert faiss_repo._doc_id_to_index == {"10": 0}


def test_faiss_repository_search_by_keyword_default_mapping(faiss_repo, tmp_path):
    faiss_repo.metadata = [{"doc_id": "10", "abstract": "abs 10", "keywords": "k10"}]
    faiss_repo._doc_id_to_index = {"10": 0}
    faiss_repo.keyword_mapping = {}

    mock_index = MagicMock()
    mock_index.ntotal = 1
    mock_index.reconstruct.return_value = np.ones(128, dtype=np.float32)
    faiss_repo.index = mock_index

    mapping_file = tmp_path / "default_mapping.json"
    mapping_file.write_text(json.dumps({"testkw": ["10"]}), encoding="utf-8")

    def mock_get(key, default=None):
        if key == "faiss_repository.mapping_path":
            return str(mapping_file)
        if key == "faiss_repository.top_k":
            return 5
        return default

    with patch("src.repositories.faiss_repository.config.get", side_effect=mock_get):
        results = faiss_repo.search_by_keyword(keyword="testkw", query="q")
        assert len(results) == 1


def test_faiss_repository_search_by_keyword_errors(faiss_repo, tmp_path):
    with pytest.raises(ValueError, match="La keyword no puede estar vacía"):
        faiss_repo.search_by_keyword(keyword="   ")

    with pytest.raises(FileNotFoundError):
        faiss_repo.search_by_keyword(keyword="test", mapping_path=tmp_path / "no.json")

    def mock_get_no_exist(key, default=None):
        if key == "faiss_repository.mapping_path":
            return str(tmp_path / "no_exists.json")
        if key == "faiss_repository.top_k":
            return 5
        return default

    with pytest.raises(FileNotFoundError, match="Carga un mapeo"):
        with patch("src.repositories.faiss_repository.config.get", side_effect=mock_get_no_exist):
            faiss_repo.search_by_keyword(keyword="test")

    faiss_repo.keyword_mapping = {"python": [1]}
    with pytest.raises(KeyError, match="La keyword 'java' no existe"):
        faiss_repo.search_by_keyword(keyword="java")