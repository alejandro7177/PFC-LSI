import logging
from dataclasses import dataclass, field
from typing import Sequence

import polars as pl
import spacy
from spellchecker import SpellChecker

from src.core.base import Filter
from src.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class KeywordNormalizerFilter(Filter):
    spacy_model: str | None = None
    use_spellcheck: bool | None = None

    # Atributos internos no inicializables mediante __init__
    nlp: spacy.language.Language = field(init=False, repr=False)
    spell: SpellChecker | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.spacy_model is None:
            self.spacy_model = config.get("keyword_normalizer.spacy_model", "en_core_web_sm")
        if self.use_spellcheck is None:
            self.use_spellcheck = config.get("keyword_normalizer.use_spellcheck", True)

        disable_components = config.get("keyword_normalizer.disable_components", ["parser", "ner"])
        logger.info(f"Cargando modelo SpaCy '{self.spacy_model}'...")
        self.nlp = spacy.load(self.spacy_model, disable=disable_components)

        if self.use_spellcheck:
            lang = config.get("keyword_normalizer.language", "en")
            logger.info("Inicializando corrector ortográfico (SpellChecker)...")
            self.spell = SpellChecker(language=lang)

    def normalize_single_keyword(self, kw: str) -> str:
        """Limpia, corrige ortografía y lematiza un solo término."""
        clean_kw = kw.strip().lower()
        if not clean_kw:
            return ""

        # 1. Corrección ortográfica (opcional)
        if self.spell:
            corrected = self.spell.correction(clean_kw)
            clean_kw = corrected if corrected else clean_kw

        # 2. Lematización con SpaCy
        doc = self.nlp(clean_kw)
        if len(doc) > 0:
            return doc[0].lemma_
        return clean_kw

    def normalize_vocabulary(self, keywords: Sequence[str]) -> list[str]:
        """Normaliza una lista o conjunto de keywords deduplicando el resultado."""
        normalized = set()
        for kw in keywords:
            lemma = self.normalize_single_keyword(kw)
            if lemma:
                normalized.add(lemma)
        return sorted(normalized)

    def process(
        self,
        df: pl.DataFrame,
        input_col: str | None = None,
        output_col: str | None = None,
    ) -> pl.DataFrame:
        """
        Procesa una columna con listas o cadenas de palabras clave en un DataFrame de Polars.
        """
        input_col = input_col or config.get("keyword_normalizer.input_col", "keywords_qwen")
        output_col = output_col or config.get("keyword_normalizer.output_col", "keywords_normalized")

        if input_col not in df.columns:
            raise ValueError(f"La columna '{input_col}' no existe en el DataFrame.")

        # Si viene como cadena separada por comas, la convertimos temporalmente a lista
        is_string_col = df[input_col].dtype == pl.Utf8
        delimiter = config.get("keyword_cleaner.delimiter", ",")
        if is_string_col:
            working_df = df.with_columns(pl.col(input_col).str.split(delimiter))
        else:
            working_df = df

        # Aplicamos la normalización lista por lista
        normalized_series = [
            self.normalize_vocabulary(row if row else [])
            for row in working_df[input_col].to_list()
        ]

        result_df = df.with_columns(
            pl.Series(name=output_col, values=normalized_series)
        )

        return result_df