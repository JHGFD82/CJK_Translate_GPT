"""PortKey pricing API integration and model catalog update functions."""

import json
import logging
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

from . import catalog as _catalog

PORTKEY_PRICING_API_BASE = "https://api.portkey.ai/model-configs/pricing"

# Maps user-facing provider prefix to PortKey's API provider slug where they differ
_PORTKEY_PROVIDER_MAP: Dict[str, str] = {
    "google": "vertex-ai",
    "mistral": "mistral-ai",
}


def _fetch_model_pricing(provider_model: str, pricing_unit: int) -> Dict[str, Any]:
    """Fetch pricing from PortKey's pricing API.

    ``provider_model`` must be in ``provider/model-name`` format.
    Prices from the API are per 100 tokens; multiply by ``pricing_unit / 100``
    to get the stored price (e.g. ×10,000 when pricing_unit=1,000,000).

    Returns a dict with at minimum 'input' and 'output' keys.
    Raises RuntimeError if the fetch fails or returns no usable pricing.
    """
    provider, model_key = provider_model.split("/", 1)
    api_provider = _PORTKEY_PROVIDER_MAP.get(provider.lower(), provider.lower())
    url = f"{PORTKEY_PRICING_API_BASE}/{api_provider}/{model_key}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=8) as response:  # nosec B310
        data = json.loads(response.read())

    pay = data.get("pay_as_you_go", {})
    input_price = float(pay.get("request_token", {}).get("price", 0))
    output_price = float(pay.get("response_token", {}).get("price", 0))

    if not (input_price > 0 and output_price > 0):
        raise RuntimeError(f"No valid pricing data for '{provider_model}' in PortKey pricing catalog")

    # PortKey stores per-100-token prices; convert to our pricing_unit
    factor = pricing_unit / 100
    return {
        "input": round(input_price * factor, 4),
        "output": round(output_price * factor, 4),
    }


def add_model_to_catalog(provider_model: str) -> Tuple[str, Dict[str, Any]]:
    """Add or update a model in the local model catalog by fetching pricing from PortKey.

    ``provider_model`` must be in ``provider/model-name`` format
    (e.g. ``openai/gpt-4o``, ``google/gemini-2.5-pro``, ``azure-ai/Llama-3.3-70B-Instruct``).
    The local catalog key will be the part after the slash (e.g. ``gpt-4o``).

    Pricing is fetched automatically from PortKey's pricing catalog.  ``supports_vision``
    defaults to ``False`` — edit ``model_catalog.json`` directly to change it.

    Returns ``(model_name, entry)``.
    """
    if "/" not in provider_model:
        raise ValueError(
            f"Model '{provider_model}' must be in 'provider/model-name' format "
            "(e.g. 'openai/gpt-4o', 'google/gemini-2.5-pro')."
        )

    _provider, model_name = provider_model.split("/", 1)

    # Load or initialize the catalog without requiring models to exist yet
    catalog_file = _catalog.get_model_catalog_path()
    if catalog_file.exists():
        try:
            with open(catalog_file, "r") as f:
                catalog = json.load(f)
            catalog.setdefault("config", {"pricing_unit": 1_000_000, "monthly_limit": 250.0})
            catalog.setdefault("models", {})
        except (json.JSONDecodeError, Exception):
            catalog = {"config": {"pricing_unit": 1_000_000, "monthly_limit": 250.0}, "models": {}}
    else:
        catalog = {"config": {"pricing_unit": 1_000_000, "monthly_limit": 250.0}, "models": {}}

    pricing_unit = catalog["config"]["pricing_unit"]

    # Preserve any existing extra fields (system_role, fixed_parameters, etc.)
    entry: Dict[str, Any] = dict(catalog["models"].get(model_name, {}))
    entry["llmprices_id"] = provider_model

    fetched = _fetch_model_pricing(provider_model, pricing_unit)
    entry["input"] = fetched["input"]
    entry["output"] = fetched["output"]
    entry["last_sync"] = datetime.now().isoformat(timespec="seconds")
    if "supports_vision" not in entry and "supports_vision" in fetched:
        entry["supports_vision"] = fetched["supports_vision"]

    entry.setdefault("supports_vision", False)

    catalog["models"][model_name] = entry
    _catalog.save_model_catalog(catalog)
    return model_name, entry


def maybe_sync_model_pricing(model: str) -> None:
    """Fetch current pricing for a model from PortKey if not synced within the last hour.

    Only runs when the model has a 'llmprices_id' field. Silently skips on any
    network or parse error so the caller is never blocked.
    """
    try:
        catalog = _catalog.load_model_catalog()
        model_entry = catalog["models"].get(model, {})
        llmprices_id = model_entry.get("llmprices_id")
        if not llmprices_id:
            return

        last_sync_str = model_entry.get("last_sync", "")
        try:
            last_sync_dt = datetime.fromisoformat(last_sync_str)
            if datetime.now() - last_sync_dt < timedelta(hours=1):
                return
        except (ValueError, TypeError):
            # Timestamp absent or in old monthly format — treat as stale and update
            pass

        pricing_unit = catalog["config"]["pricing_unit"]
        fetched = _fetch_model_pricing(llmprices_id, pricing_unit)
        now_iso = datetime.now().isoformat(timespec="seconds")
        catalog["models"][model]["input"] = fetched["input"]
        catalog["models"][model]["output"] = fetched["output"]
        catalog["models"][model]["last_sync"] = now_iso
        _catalog.save_model_catalog(catalog)
        logging.info(
            f"Synced pricing for {model}: "
            f"input=${catalog['models'][model]['input']}, "
            f"output=${catalog['models'][model]['output']}"
        )
    except Exception as e:
        logging.warning(f"Could not sync pricing for {model}: {e}")
