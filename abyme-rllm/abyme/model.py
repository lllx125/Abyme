from abc import ABC, abstractmethod
import os
from openai import OpenAI
from typing import  TYPE_CHECKING
from dotenv import load_dotenv

# Lazy import for HuggingFaceModel - only loaded when accessed
# This allows lightweight usage with OpenAI/DeepSeek models without PyTorch dependencies
if TYPE_CHECKING:
    from .pytorch_modules import HuggingFaceModel

# Load environment variables from .env file
load_dotenv()


def __getattr__(name):
    """Lazy import mechanism for HuggingFaceModel to avoid PyTorch dependency."""
    if name == "HuggingFaceModel":
        from .pytorch_modules import HuggingFaceModel
        return HuggingFaceModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Model(ABC):
    """Abstract base class for language models."""

    @abstractmethod
    def generate(self, prompt: str, max_attempt: int) -> str:
        """
        Generate a response from the model with retry capability.

        Args:
            prompt: The input prompt string
            max_attempt: Maximum number of retry attempts on failure

        Returns:
            The generated response string

        Raises:
            Exception: If all attempts fail
        """
        pass


class DeepSeekModel(Model):
    """DeepSeek model implementation using OpenAI-compatible API."""

    def __init__(
        self,
        reasoning: bool = False,
        system_prompt: str = "You are a helpful AI assistant."
    ):
        """
        Initialize DeepSeek model.

        Args:
            reasoning: If True, uses deepseek-reasoner; otherwise uses deepseek-chat
            system_prompt: System prompt to use for all generations
        """
        self.model_name = "deepseek-reasoner" if reasoning else "deepseek-chat"
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = "https://api.deepseek.com"
        self.system_prompt = system_prompt

        # Initialize OpenAI client for DeepSeek
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate(self, prompt: str, max_attempt: int = 1) -> str:
        """
        Generate a response using the DeepSeek API with retry mechanism.

        Args:
            prompt: The user prompt
            max_attempt: Maximum number of retry attempts on failure (default: 1)

        Returns:
            The generated response

        Raises:
            Exception: If all attempts fail
        """
        last_error = None
        for _ in range(max_attempt):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                content = response.choices[0].message.content
                return content if content is not None else ""
            except Exception as e:
                last_error = e
                continue

        raise Exception(f"All {max_attempt} attempts failed. Last error: {last_error}")


class GPTModel(Model):
    """GPT-5 model implementation using the new responses API."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful AI assistant."
    ):
        """
        Initialize GPT-5 model.

        Args:
            system_prompt: System prompt to use for all generations
        """
        self.system_prompt = system_prompt
        # OpenAI client automatically loads API key from OPENAI_API_KEY environment variable
        self.client = OpenAI()

    def generate(self, prompt: str, max_attempt: int = 1) -> str:
        """
        Generate a response using the GPT-5 responses API with retry mechanism.

        Args:
            prompt: The user prompt
            max_attempt: Maximum number of retry attempts on failure (default: 1)

        Returns:
            The generated response

        Raises:
            Exception: If all attempts fail
        """
        last_error = None
        for _ in range(max_attempt):
            try:
                response = self.client.responses.create(
                    model="gpt-5",
                    input=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                # Extract content from the response
                content = response.output_text if hasattr(response, 'output_text') else str(response)
                return content if content is not None else ""
            except Exception as e:
                last_error = e
                continue

        raise Exception(f"All {max_attempt} attempts failed. Last error: {last_error}")
    
class ErrorGuardModel(Model):
    def generate(self, prompt: str, max_retry:int) -> str:
        return "You reached recursion limit, you must solve this problem your self and delegate no further"