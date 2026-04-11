"""Custom prompt service for direct AI interaction."""

import logging
from typing import Any, Optional
from collections.abc import Iterator as ABCIterator

from portkey_ai import Portkey

from ..models import (
    resolve_model, get_model_system_role,
    model_uses_max_completion_tokens, model_has_fixed_parameters,
    maybe_sync_model_pricing,
)
from ..tracking.token_tracker import TokenTracker
from .api_errors import handle_common_api_errors
from .constants import MAX_RETRIES, BASE_RETRY_DELAY

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# API parameters for general-purpose prompts
PROMPT_MAX_TOKENS: int = 4000
PROMPT_TEMPERATURE: float = 0.7
PROMPT_TOP_P: float = 1.0


class PromptService:
    """Sends custom prompts to the AI model and returns the response."""

    def __init__(
        self,
        api_key: str,
        professor: Optional[str] = None,
        token_tracker: Optional[TokenTracker] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key
        self.professor = professor
        self.custom_model = model
        self.client = Portkey(api_key=api_key)
        self.token_tracker = (
            token_tracker
            if token_tracker is not None
            else TokenTracker(professor=professor or "")
        )

    def _get_model(self) -> str:
        """Resolve model, syncing pricing if needed."""
        model = resolve_model(requested_model=self.custom_model)
        maybe_sync_model_pricing(model)
        return model

    def _call_api(self, model: str, system_role: str, system_prompt: str, user_prompt: str) -> Any:
        """Call the API with parameters appropriate for the given model."""
        messages = [
            {"role": system_role, "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        use_completion_tokens = model_uses_max_completion_tokens(model)
        fixed_params = model_has_fixed_parameters(model)
        if use_completion_tokens and fixed_params:
            return self.client.chat.completions.create(  # type: ignore[misc]
                model=model,
                max_completion_tokens=PROMPT_MAX_TOKENS,
                stream=False,
                messages=messages,
            )
        if use_completion_tokens:
            return self.client.chat.completions.create(  # type: ignore[misc]
                model=model,
                temperature=PROMPT_TEMPERATURE,
                max_completion_tokens=PROMPT_MAX_TOKENS,
                top_p=PROMPT_TOP_P,
                stream=False,
                messages=messages,
            )
        return self.client.chat.completions.create(  # type: ignore[misc]
            model=model,
            temperature=PROMPT_TEMPERATURE,
            max_tokens=PROMPT_MAX_TOKENS,
            top_p=PROMPT_TOP_P,
            stream=False,
            messages=messages,
        )

    def build_prompts(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) without calling the API.

        Used by --dry-run mode to preview what would be sent to the model.
        """
        return system_prompt or DEFAULT_SYSTEM_PROMPT, user_prompt

    def send_prompt(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Send a custom prompt and return the response text."""
        model = self._get_model()
        system_role = get_model_system_role(model)
        effective_system = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

        logging.info(f"Sending custom prompt to model: {model}")
        try:
            response = self._call_api(model, system_role, effective_system, user_prompt)
        except Exception as e:
            handle_common_api_errors(e, model)
            raise

        assert not isinstance(response, ABCIterator), "Unexpected stream response received."

        if (
            response.usage
            and response.usage.prompt_tokens is not None
            and response.usage.completion_tokens is not None
            and response.usage.total_tokens is not None
        ):
            usage = self.token_tracker.record_usage(
                model=response.model or model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                requested_model=model,
            )
            logging.info(
                f"Tokens — prompt: {response.usage.prompt_tokens}, "
                f"completion: {response.usage.completion_tokens}, "
                f"cost: ${usage.total_cost:.4f}"
            )
        else:
            logging.warning("No token usage information available in response.")

        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            if content is not None and isinstance(content, str):
                return content

        return ""
