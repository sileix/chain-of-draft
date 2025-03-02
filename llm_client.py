from anthropic import Anthropic
from openai import OpenAI


class LLMClient:
    def __init__(self):
        self.openai_client = OpenAI()
        self.anthropic_client = Anthropic()

    def request(
        self,
        payload: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[str, int]:
        if model.startswith("gpt"):
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
        else:
            raise ValueError("Unrecognized model name: " + model)
        return response, token_count


if __name__ == "__main__":
    llm = LLMClient()
    response, count = llm.request("hello", "gpt-4o")
    print(response, count)
