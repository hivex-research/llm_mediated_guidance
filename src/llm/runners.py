from abc import ABC, abstractmethod
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class LLMRunner(ABC):
    """Abstract base class for LLM providers."""

    API_KEY: str
    MODEL: str

    @abstractmethod
    def __init__(self):
        """Initialize the LLM client."""
        return NotImplementedError

    @abstractmethod
    def run(prompt: str) -> str:
        """Run the LLM with the given prompt and return the response."""
        return NotImplementedError


class DeepSeekReasoningRunner(LLMRunner):
    API_KEY = os.getenv("DEEPSEEK")
    MODEL = "deepseek-reasoner"  # DeepSeek-V3.1-Think

    def __init__(self):
        self.client = OpenAI(api_key=self.API_KEY, base_url="https://api.deepseek.com")

    def run(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": "Be as concise as possible."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
