"""
Shared constants for all service modules (rate limiting, retry behaviour).
"""

# Rate limiting and retry configuration
PAGE_DELAY_SECONDS: float = 3.0  # Delay between pages to prevent content filter triggers
MAX_RETRIES: int = 10            # Maximum retries for content filter / transient errors
BASE_RETRY_DELAY: float = 3.0   # Base delay (seconds) for exponential backoff
