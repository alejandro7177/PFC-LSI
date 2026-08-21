import json
import re
from typing import Generator
import spacy
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from src.core.base import Filter
from src.core.config import config

spacy_model = config.get("keyword_extractor.spacy_model", "en_core_web_sm")
nlp = spacy.load(spacy_model, disable=["textcat"])


class KeywordExtractorFilter(Filter):
    def __init__(self, vocab_path: str | None = None, model_name: str | None = None):
        self.vocab_path = vocab_path or config.get("keyword_extractor.vocab_path", "data/artifacts/vocabulary.json")
        self.model_name = model_name or config.get("keyword_extractor.model_name", "Qwen/Qwen3-Embedding-0.6B")
        self._kw_model = None

    @property
    def kw_model(self) -> KeyBERT:
        if self._kw_model is None:
            embedding = SentenceTransformer(self.model_name)
            self._kw_model = KeyBERT(model=embedding)
        return self._kw_model

    def _extract_candidates(self, text: str) -> list[str]:
        doc = nlp(text)
        candidates = set()
        for chunk in doc.noun_chunks:
            c = re.sub(r"\s+", " ", chunk.text.strip().lower())
            if 1 <= len(c.split()) <= 5:
                candidates.add(c)
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop and not token.is_punct and len(token.text) >= 3:
                candidates.add(token.lemma_.lower())
        return sorted([c for c in candidates if len(c) >= 3])

    def process(self, stream: Generator[dict, None, None]) -> Generator[dict, None, None]:
        top_n = config.get("keyword_extractor.top_n", 4)
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocabulary = json.load(f)
        vectorizer = CountVectorizer(vocabulary=vocabulary, ngram_range=(1, 1))

        for item in stream:
            abstract = item.get("abstract", "")
            if abstract:
                keywords = self.kw_model.extract_keywords(
                    abstract,
                    vectorizer=vectorizer,
                    top_n=top_n,
                    stop_words="english"
                )
                item["keywords"] = ", ".join([kw[0] for kw in keywords])
            else:
                item["keywords"] = ""
            yield item