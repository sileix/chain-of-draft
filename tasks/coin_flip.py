import json
import os
import random
from typing import List

from names_dataset import NameDataset

from llm_client import LLMClient
from tasks.base import Task
from utils import Example

TEST_SET_SIZE = 250


class CoinFlip(Task):
    def __init__(self, flip_count: int = 4):
        super().__init__("coin_flip", LLMClient())
        self.flip_count = flip_count
        self.data_file = f"./data/coin_flip_{self.flip_count}.json"

    def get_top_names(self) -> List[str]:
        nd = NameDataset()
        top_names = nd.get_top_names(500, True, "US")
        return top_names["US"]["M"] + top_names["US"]["F"]

    def synthesize_example(self, names: List[str]) -> dict:
        statements = ["A coin is heads up."]
        heads_up = True
        for name in names:
            flip = random.choice([True, False])
            if flip:
                statements.append(f"{name} flips the coin.")
                heads_up = not heads_up
            else:
                statements.append(f"{name} does not flip the coin.")
        return {
            "question": " ".join(statements + ["Is the coin still heads up?"]),
            "answer": "Yes" if heads_up else "No",
        }

    def synthesize_data(self):
        top_names = self.get_top_names()
        examples = []
        while len(examples) < TEST_SET_SIZE:
            names = random.sample(top_names, self.flip_count)
            examples.append(self.synthesize_example(names))
        with open(self.data_file, "w") as f:
            json.dump({"flip_count": self.flip_count, "examples": examples}, f)

    def load_data(self) -> List[Example]:
        if not os.path.exists(self.data_file):
            self.synthesize_data()
        data = []
        with open(self.data_file) as f:
            for example in json.load(f)["examples"]:
                data.append(Example.model_validate(example))
        return data

    def extract_answer(self, raw_response: str) -> str:
        raw_response = raw_response.strip()
        try:
            if raw_response.lower() == "yes":
                return "Yes"
            if raw_response.lower() == "no":
                return "No"
            raise ValueError()
        except ValueError:
            pass

        try:
            answer = raw_response.split("####")[1]
            return self.extract_answer(answer)
        except Exception:
            pass

        print("Failed to extract answer from the following response:")
        print(raw_response)
        return "N/A"

    def equal(self, predicted_answer: str, expected_answer: str) -> bool:
        return predicted_answer == expected_answer
