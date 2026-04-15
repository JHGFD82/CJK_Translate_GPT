"""PortKey pricing API integration and model catalog update functions."""

import json
import logging
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

from . import catalog as _catalog

# In-memory cache: model name → last-synced datetime.
# Prevents repeated catalog disk reads when many workers share the same model.
_sync_cache: Dict[str, datetime] = {}

PORTKEY_PRICING_API_BASE = "https://api.portkey.ai/model-configs/pricing"


def _fetch_model_pricing(provider_model: str, pricing_unit: int) -> Dict[str, Any]:
    """Fetch pricing from PortKey's pricing API.

    ``provider_model`` must be in ``provider/model-name`` format.
    Prices from the API are per 100 tokens; multiply by ``pricing_unit / 100``
    to get the stored price (e.g. ×10,000 when pricing_unit=1,000,000).

    Returns a dict with at minimum 'input' and 'output' keys.
    Raises RuntimeError if the fetch fails or returns no usable pricing.
    """
    provider, model_key = provider_model.split("/", 1)
    catalog = _catalog.load_model_catalog()
    provider_map = catalog.get("config", {}).get("provider_map", {})
    api_provider = provider_map.get(provider.lower(), provider.lower())
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
    entry["portkey_id"] = provider_model

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

    Only runs when the model has a 'portkey_id' field. Silently skips on any
    network or parse error so the caller is never blocked.
    """
    # Fast in-memory check — avoids disk reads when workers share the same model
    cached_at = _sync_cache.get(model)
    if cached_at is not None and datetime.now() - cached_at < timedelta(hours=1):
        return

    try:
        catalog = _catalog.load_model_catalog()
        model_entry = catalog["models"].get(model, {})
        portkey_id = model_entry.get("portkey_id")
        if not portkey_id:
            _sync_cache[model] = datetime.now()  # no portkey_id — nothing to sync
            return

        last_sync_str = model_entry.get("last_sync", "")
        try:
            last_sync_dt = datetime.fromisoformat(last_sync_str)
            if datetime.now() - last_sync_dt < timedelta(hours=1):
                _sync_cache[model] = last_sync_dt
                return
        except (ValueError, TypeError):
            # Timestamp absent or in old monthly format — treat as stale and update
            pass

        pricing_unit = catalog["config"]["pricing_unit"]
        fetched = _fetch_model_pricing(portkey_id, pricing_unit)
        now_dt = datetime.now()
        now_iso = now_dt.isoformat(timespec="seconds")
        catalog["models"][model]["input"] = fetched["input"]
        catalog["models"][model]["output"] = fetched["output"]
        catalog["models"][model]["last_sync"] = now_iso
        _catalog.save_model_catalog(catalog)
        _sync_cache[model] = now_dt
        logging.info(
            f"Synced pricing for {model}: "
            f"input=${catalog['models'][model]['input']}, "
            f"output=${catalog['models'][model]['output']}"
        )
    except Exception as e:
        logging.warning(f"Could not sync pricing for {model}: {e}")
