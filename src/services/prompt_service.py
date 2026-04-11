"""Custom prompt service for direct AI interaction."""

import logging
from typing import Any, Optional

from ..models import (
    resolve_model, get_model_system_role,
    maybe_sync_model_pricing,
)
from ..tracking.token_tracker import TokenTracker
from .api_errors import handle_api_errors
from .base_service import BaseService


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# API parameters for general-purpose prompts
PROMPT_MAX_TOKENS: int = 4000
PROMPT_TEMPERATURE: float = 0.7
PROMPT_TOP_P: float = 1.0


class PromptService(BaseService):
    """Sends custom prompts to the AI model and returns the response."""

    def __init__(
        self,
        api_key: str,
        professor: Optional[str] = None,
        token_tracker: Optional[TokenTracker] = None,
        model: Optional[str] = None,
    ):
        super().__init__(api_key, professor, token_tracker, None, model)

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
        return self._create_completion(
            model, messages, PROMPT_MAX_TOKENS,
            temperature=PROMPT_TEMPERATURE, top_p=PROMPT_TOP_P,
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
            handle_api_errors(e, model)
            raise

        self._record_response_usage(response, model)

        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            if content is not None and isinstance(content, str):
                return content

        return ""
