import os
import zhipuai

from anthropic import Anthropic
from openai import OpenAI


class LLMClient:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = Anthropic()
        self.local_client = None
        self.deepinfra_client = None
        # Initialize Zhipu client (using Zhipu SDK)
        self.zhipu_client = None

    def request(
        self,
        payload: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        if model.startswith("gpt"):
            if self.openai_client is None:
                try:
                    self.openai_client = OpenAI()
                except Exception as e:
                    print(f"OpenAI initialization error: {e}")
                    return "", 0  # Handle the error appropriately
            completion = self.openai_client.chat.completions.create(
                messages=[{"role": "user", "content": payload}],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = completion.choices[0].message.content
            token_count = completion.usage.completion_tokens
        elif model.startswith("claude"):
            message = self.anthropic_client.messages.create(
                messages=[{"role": "user", "content": payload}],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = message.content[0].text
            token_count = message.usage.output_tokens
        elif model.startswith("deepinfra:"):
            if self.deepinfra_client is None:
                try:
                    self.deepinfra_client = OpenAI(
                        api_key=os.getenv("DEEPINFRA_API_KEY"),
                        base_url="https://api.deepinfra.com/v1/openai",
                    )
                except Exception as e:
                    print(f"DeepInfra initialization error: {e}")
                    return "", 0  # Handle the error appropriately
            completion = self.deepinfra_client.chat.completions.create(
                messages=[{"role": "user", "content": payload}],
                model=model.replace("deepinfra:", ""),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = completion.choices[0].message.content
            token_count = completion.usage.completion_tokens
        elif model.startswith("local:"):
            if self.local_client is None:
                try:
                    self.local_client = OpenAI(base_url="http://127.0.0.1:8000/v1")
                except Exception as e:
                    print(f"Local client initialization error: {e}")
                    return "", 0  # Handle the error appropriately
            completion = self.local_client.chat.completions.create(
                messages=[{"role": "user", "content": payload}],
                model=model.replace("local:", ""),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = completion.choices[0].message.content
            token_count = completion.usage.completion_tokens
        # Add Zhipu support (using Zhipu SDK)
        else:
            if self.zhipu_client is None:
                try:
                    self.zhipu_client = zhipuai.ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
                except Exception as e:
                    print(f"Zhipu client initialization error: {e}")
                    return "", 0  # Handle the error appropriately
            try:
                response = self.zhipu_client.chat.completions.create(
                    model=model,  # Directly use the model name
                    messages=[{"role": "user", "content": payload}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response_text = response.choices[0].message.content
                token_count = response.usage.total_tokens  # Need to confirm the key name
                return response_text, token_count
            except Exception as e:
                print(f"Zhipu API error: {e}")
                return "", 0  # Handle the error appropriately
