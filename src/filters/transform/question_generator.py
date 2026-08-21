import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.base import Filter
from src.core.config import config

logger = logging.getLogger(__name__)


def extract_keyword_groups(
    csv_path: str | Path | None = None,
    kw_column: str | None = None,
    num_questions: int | None = None,
    group_size: int | None = None,
    random_state: int | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Lee el CSV y genera un stream de grupos de abstracts que comparten keyword."""
    csv_path = Path(csv_path or config.get("paths.dataset_csv", "data/dataset.csv"))
    kw_column = kw_column or config.get("question_generator.kw_column", "kw_v1_df30")
    num_questions = num_questions or config.get("question_generator.num_questions", 1000)
    group_size = group_size or config.get("question_generator.group_size", 5)
    random_state = random_state if random_state is not None else config.get("question_generator.random_state", 42)

    df = pd.read_csv(csv_path)

    kw_to_indices = defaultdict(list)
    for idx, row in df.iterrows():
        kw_str = str(row[kw_column]) if pd.notna(row[kw_column]) else ""
        if not kw_str or not str(row["abstract"]).strip():
            continue
        keywords = [k.strip() for k in kw_str.split(",") if k.strip()]
        for kw in set(keywords):
            kw_to_indices[kw].append(idx)

    valid_kw_indices = {
        k: v for k, v in kw_to_indices.items() if len(v) >= group_size
    }

    if not valid_kw_indices:
        raise ValueError(
            f"No se encontraron keywords con al menos {group_size} abstracts."
        )

    candidate_groups = []
    for kw, indices in valid_kw_indices.items():
        indices_copy = list(indices)
        if random_state is not None:
            random.Random(random_state + hash(kw) % 10000).shuffle(indices_copy)

        for i in range(0, len(indices_copy) - group_size + 1, group_size):
            candidate_groups.append((kw, indices_copy[i : i + group_size]))

    if random_state is not None:
        random.Random(random_state).shuffle(candidate_groups)

    selected_groups = candidate_groups[:num_questions]

    for kw, idxs in selected_groups:
        doc_ids = []
        abstracts = []
        for i in idxs:
            df_row = df.loc[i]
            doc_ids.append(df_row["doc_id"] if "doc_id" in df_row else i)
            abstracts.append(str(df_row["abstract"]))

        yield {
            "keyword": kw,
            "doc_ids": doc_ids,
            "abstracts": abstracts,
        }


@dataclass
class QuestionGeneratorFilter(Filter):
    model_name: str | None = None
    prompt_path: str | None = None
    max_new_tokens: int | None = None
    cuda_gpu: str | None = None
    batch_size: int | None = None

    def __post_init__(self):
        if self.model_name is None:
            self.model_name = config.get("question_generator.model_name", "google/gemma-4-12B-it")
        if self.prompt_path is None:
            self.prompt_path = config.get("question_generator.prompt_path", "prompts/question_generation_v1.md")
        if self.max_new_tokens is None:
            self.max_new_tokens = config.get("question_generator.max_new_tokens", 256)
        if self.cuda_gpu is None:
            self.cuda_gpu = str(config.get("question_generator.cuda_gpu", "1"))
        if self.batch_size is None:
            self.batch_size = config.get("question_generator.batch_size", 16)

        logger.info(f"Cargando tokenizer de {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        logger.info(f"Cargando modelo {self.model_name} en cuda:{self.cuda_gpu}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{self.cuda_gpu}"},
        )

        self.prompt_template = Path(self.prompt_path).read_text(encoding="utf-8")

    def _build_prompt_5(self, abstracts: list[str]) -> str:
        prompt_str = self.prompt_template
        if len(abstracts) < 5:
            abstracts = (list(abstracts) + [""] * 5)[:5]

        for i in range(1, 6):
            placeholder_double = f"{{{{abstract_{i}}}}}"
            placeholder_single = f"{{abstract_{i}}}"
            abs_text = abstracts[i - 1].strip()
            if placeholder_double in prompt_str:
                prompt_str = prompt_str.replace(placeholder_double, abs_text)
            elif placeholder_single in prompt_str:
                prompt_str = prompt_str.replace(placeholder_single, abs_text)
        return prompt_str

    def _parse_response(self, response: str) -> dict[str, Any]:
        clean_response = response.strip()
        if clean_response.startswith("```"):
            clean_response = re.sub(r"^```(?:json)?\n?", "", clean_response)
            clean_response = re.sub(r"\n?```$", "", clean_response).strip()

        try:
            return json.loads(clean_response)
        except json.JSONDecodeError:
            return {"raw_response": response}

    def generate_batch_5(self, groups_of_5: list[list[str]]) -> list[dict[str, Any]]:
        if not groups_of_5:
            return []

        batch_messages = [
            [{"role": "user", "content": self._build_prompt_5(group)}]
            for group in groups_of_5
        ]

        inputs = self.tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[1]
        decoded_responses = self.tokenizer.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=True,
        )

        return [self._parse_response(resp) for resp in decoded_responses]

    def process(self, stream: Generator[dict[str, Any], None, None]) -> Generator[dict[str, Any], None, None]:
        batch = []
        for item in stream:
            batch.append(item)
            if len(batch) >= self.batch_size:
                abstract_groups = [b["abstracts"] for b in batch]
                responses = self.generate_batch_5(abstract_groups)

                for b_item, resp in zip(batch, responses):
                    b_item["question"] = resp.get("question", resp.get("raw_response", ""))
                    yield b_item
                batch = []

        if batch:
            abstract_groups = [b["abstracts"] for b in batch]
            responses = self.generate_batch_5(abstract_groups)
            for b_item, resp in zip(batch, responses):
                b_item["question"] = resp.get("question", resp.get("raw_response", ""))
                yield b_item