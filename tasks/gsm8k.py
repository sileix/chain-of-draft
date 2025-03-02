from typing import List

from datasets import load_dataset

from llm_client import LLMClient
from tasks.base import Task
from utils import Example, extract_number_from_string


class GSM8K(Task):
    def __init__(self):
        super().__init__("gsm8k", LLMClient())

    def load_data(self) -> List[Example]:
        data = []
        for example in load_dataset("openai/gsm8k", "main", split="test"):
            data.append(Example.model_validate(example))
        return data

    def extract_answer(self, raw_response: str) -> str:
        if "####" in raw_response:
            raw_response = raw_response.split("####")[1]
        raw_response = raw_response.strip().replace(",", "").replace("$", "").replace("%", "")
        return raw_response

    def equal(self, predicted_answer: str, expected_answer: str) -> bool:
        if predicted_answer == expected_answer:
            return True
        predicted_answer = str(extract_number_from_string(predicted_answer))
        if predicted_answer == expected_answer:
            return True
        try:
            if float(predicted_answer) == float(expected_answer):
                return True
        except Exception:
            return False
        return False
