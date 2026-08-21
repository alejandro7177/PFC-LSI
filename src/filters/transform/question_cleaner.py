import json
import re
from typing import Generator
from src.core.base import Filter
from src.core.config import config


class QuestionCleanerFilter(Filter):
    def __init__(self, input_col: str | None = None, output_col: str | None = None):
        self.input_col = input_col or config.get("question_cleaner.input_col", "question_gemma4_12b")
        self.output_col = output_col or config.get("question_cleaner.output_col", "question_gemma4_12b_clean")

    def _clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        cleaned_str = text.strip()

        if "```" in cleaned_str or "'''" in cleaned_str:
            cleaned_str = re.sub(r"^(?:```|''')(?:json)?\s*", "", cleaned_str, flags=re.IGNORECASE)
            cleaned_str = re.sub(r"\s*(?:```|''')$", "", cleaned_str).strip()

        try:
            parsed = json.loads(cleaned_str)
            if isinstance(parsed, dict) and "question" in parsed:
                return parsed["question"].strip()
        except (json.JSONDecodeError, TypeError):
            pass

        match = re.search(r'"question"\s*:\s*"(.*?)"(?:\s*\}|\s*,|\s*$)', cleaned_str, re.DOTALL)
        if match:
            return match.group(1).strip()

        return cleaned_str

    def process(self, stream: Generator[dict, None, None]) -> Generator[dict, None, None]:
        for item in stream:
            if self.input_col in item:
                item[self.output_col] = self._clean_text(item[self.input_col])
            yield item