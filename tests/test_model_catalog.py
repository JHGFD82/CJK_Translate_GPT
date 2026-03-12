"""
Tests for model-catalog functions in src/config.py:
  - get_model_catalog_path
  - load_model_catalog
  - get_available_models
  - get_model_pricing
  - get_pricing_unit
  - get_monthly_limit
  - model_supports_vision
  - get_vision_capable_models
  - get_model_system_role
  - model_uses_max_completion_tokens
  - model_has_fixed_parameters
  - get_model_max_completion_tokens
  - resolve_model
  - save_model_catalog
"""

import json

import pytest

import src.config as config_module
from src.config import (
    get_available_models,
    get_model_catalog_path,
    get_model_max_completion_tokens,
    get_model_pricing,
    get_model_system_role,
    get_monthly_limit,
    get_pricing_unit,
    get_vision_capable_models,
    load_model_catalog,
    model_has_fixed_parameters,
    model_supports_vision,
    model_uses_max_completion_tokens,
    resolve_model,
    save_model_catalog,
)

# ---------------------------------------------------------------------------
# Shared test catalog — used by every test that mocks load_model_catalog
# ---------------------------------------------------------------------------

SAMPLE_CATALOG = {
    "config": {
        "pricing_unit": 1_000_000,
        "monthly_limit": 250.0,
    },
    "models": {
        "gpt-5": {
            "input": 1.38,
            "output": 11.0,
            "supports_vision": True,
            "system_role": "developer",
            "use_max_completion_tokens": True,
            "fixed_parameters": True,
            "max_completion_tokens": 16000,
        },
        "gpt-4o": {
            "input": 2.75,
            "output": 11.0,
            "supports_vision": True,
        },
        "gpt-4o-mini": {
            "input": 0.165,
            "output": 0.66,
            "supports_vision": True,
        },
        "text-only-model": {
            "input": 0.10,
            "output": 0.30,
            "supports_vision": False,
        },
    },
}


@pytest.fixture()
def mock_catalog(monkeypatch):
    """Patch load_model_catalog to return SAMPLE_CATALOG without hitting disk."""
    monkeypatch.setattr(config_module, "load_model_catalog", lambda: SAMPLE_CATALOG)


# ---------------------------------------------------------------------------
# get_model_catalog_path
# ---------------------------------------------------------------------------

class TestGetModelCatalogPath:

    def test_returns_path_ending_in_catalog_filename(self):
        path = get_model_catalog_path()
        assert path.name == "model_catalog.json"

    def test_path_is_inside_src_directory(self):
        path = get_model_catalog_path()
        assert path.parent.name == "src"


# ---------------------------------------------------------------------------
# load_model_catalog
# ---------------------------------------------------------------------------

class TestLoadModelCatalog:

    def test_missing_file_raises_file_not_found(self, monkeypatch, tmp_path):
        missing = tmp_path / "nonexistent.json"
        monkeypatch.setattr(config_module, "get_model_catalog_path", lambda: missing)
        with pytest.raises(FileNotFoundError):
            load_model_catalog()

    def test_invalid_json_raises_value_error(self, monkeypatch, tmp_path):
        bad_file = tmp_path / "model_catalog.json"
        bad_file.write_text("{ not valid json }")
        monkeypatch.setattr(config_module, "get_model_catalog_path", lambda: bad_file)
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_model_catalog()

    def test_missing_config_section_raises_value_error(self, monkeypatch, tmp_path):
        catalog = tmp_path / "model_catalog.json"
        catalog.write_text(json.dumps({"models": {"gpt-4o": {}}}))
        monkeypatch.setattr(config_module, "get_model_catalog_path", lambda: catalog)
        with pytest.raises(ValueError, match="'config' section"):
            load_model_catalog()

    def test_missing_models_section_raises_value_error(self, monkeypatch, tmp_path):
        catalog = tmp_path / "model_catalog.json"
        catalog.write_text(json.dumps({"config": {"pricing_unit": 1000000, "monthly_limit": 250.0}}))
        monkeypatch.setattr(config_module, "get_model_catalog_path", lambda: catalog)
        with pytest.raises(ValueError, match="'models' section"):
            load_model_catalog()

    def test_empty_models_section_raises_value_error(self, monkeypatch, tmp_path):
        catalog = tmp_path / "model_catalog.json"
        catalog.write_text(json.dumps({"config": {}, "models": {}}))
        monkeypatch.setattr(config_module, "get_model_catalog_path", lambda: catalog)
        with pytest.raises(ValueError, match="no models"):
            load_model_catalog()

    def test_valid_catalog_returns_dict(self, monkeypatch, tmp_path):
        catalog = tmp_path / "model_catalog.json"
        catalog.write_text(json.dumps(SAMPLE_CATALOG))
        monkeypatch.setattr(config_module, "get_model_catalog_path", lambda: catalog)
        result = load_model_catalog()
        assert result["config"]["pricing_unit"] == 1_000_000
        assert "gpt-4o" in result["models"]


# ---------------------------------------------------------------------------
# get_available_models
# ---------------------------------------------------------------------------

class TestGetAvailableModels:

    def test_returns_all_model_keys(self, mock_catalog):
        models = get_available_models()
        assert set(models) == {"gpt-5", "gpt-4o", "gpt-4o-mini", "text-only-model"}

    def test_returns_list(self, mock_catalog):
        assert isinstance(get_available_models(), list)


# ---------------------------------------------------------------------------
# get_model_pricing
# ---------------------------------------------------------------------------

class TestGetModelPricing:

    def test_known_model_returns_pricing(self, mock_catalog):
        pricing = get_model_pricing("gpt-4o")
        assert pricing["input"] == 2.75
        assert pricing["output"] == 11.0

    def test_unknown_model_falls_back_to_gpt4o_mini(self, mock_catalog):
        # DEFAULT_FALLBACK_MODEL = "gpt-4o-mini" which is in SAMPLE_CATALOG
        pricing = get_model_pricing("unknown-model")
        assert pricing["input"] == pytest.approx(0.165)

    def test_unknown_model_no_fallback_raises(self, monkeypatch):
        catalog_no_mini = {
            "config": {"pricing_unit": 1_000_000, "monthly_limit": 250.0},
            "models": {"gpt-4o": {"input": 2.75, "output": 11.0, "supports_vision": True}},
        }
        monkeypatch.setattr(config_module, "load_model_catalog", lambda: catalog_no_mini)
        with pytest.raises(ValueError, match="not found in model catalog"):
            get_model_pricing("mystery-model")


# ---------------------------------------------------------------------------
# get_pricing_unit / get_monthly_limit
# ---------------------------------------------------------------------------

class TestConfigValues:

    def test_get_pricing_unit(self, mock_catalog):
        assert get_pricing_unit() == 1_000_000

    def test_get_monthly_limit(self, mock_catalog):
        assert get_monthly_limit() == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# model_supports_vision
# ---------------------------------------------------------------------------

class TestModelSupportsVision:

    def test_vision_model_returns_true(self, mock_catalog):
        assert model_supports_vision("gpt-4o") is True

    def test_non_vision_model_returns_false(self, mock_catalog):
        assert model_supports_vision("text-only-model") is False

    def test_unknown_model_returns_false(self, mock_catalog):
        assert model_supports_vision("ghost-model") is False

    def test_gpt5_supports_vision(self, mock_catalog):
        assert model_supports_vision("gpt-5") is True


# ---------------------------------------------------------------------------
# get_vision_capable_models
# ---------------------------------------------------------------------------

class TestGetVisionCapableModels:

    def test_returns_only_vision_models(self, mock_catalog):
        vision_models = get_vision_capable_models()
        assert "text-only-model" not in vision_models
        assert set(vision_models) == {"gpt-5", "gpt-4o", "gpt-4o-mini"}

    def test_returns_list(self, mock_catalog):
        assert isinstance(get_vision_capable_models(), list)


# ---------------------------------------------------------------------------
# get_model_system_role
# ---------------------------------------------------------------------------

class TestGetModelSystemRole:

    def test_reasoning_model_returns_developer(self, mock_catalog):
        assert get_model_system_role("gpt-5") == "developer"

    def test_standard_model_defaults_to_system(self, mock_catalog):
        # gpt-4o has no system_role key in SAMPLE_CATALOG → defaults to "system"
        assert get_model_system_role("gpt-4o") == "system"

    def test_unknown_model_defaults_to_system(self, mock_catalog):
        assert get_model_system_role("nonexistent-model") == "system"


# ---------------------------------------------------------------------------
# model_uses_max_completion_tokens
# ---------------------------------------------------------------------------

class TestModelUsesMaxCompletionTokens:

    def test_reasoning_model_returns_true(self, mock_catalog):
        assert model_uses_max_completion_tokens("gpt-5") is True

    def test_standard_model_returns_false(self, mock_catalog):
        assert model_uses_max_completion_tokens("gpt-4o") is False

    def test_unknown_model_returns_false(self, mock_catalog):
        assert model_uses_max_completion_tokens("mystery") is False


# ---------------------------------------------------------------------------
# model_has_fixed_parameters
# ---------------------------------------------------------------------------

class TestModelHasFixedParameters:

    def test_reasoning_model_returns_true(self, mock_catalog):
        assert model_has_fixed_parameters("gpt-5") is True

    def test_standard_model_returns_false(self, mock_catalog):
        assert model_has_fixed_parameters("gpt-4o") is False

    def test_unknown_model_returns_false(self, mock_catalog):
        assert model_has_fixed_parameters("mystery") is False


# ---------------------------------------------------------------------------
# get_model_max_completion_tokens
# ---------------------------------------------------------------------------

class TestGetModelMaxCompletionTokens:

    def test_model_with_override_returns_override(self, mock_catalog):
        assert get_model_max_completion_tokens("gpt-5", default=4096) == 16000

    def test_model_without_override_returns_default(self, mock_catalog):
        # gpt-4o has no max_completion_tokens in SAMPLE_CATALOG
        assert get_model_max_completion_tokens("gpt-4o", default=4096) == 4096

    def test_unknown_model_returns_default(self, mock_catalog):
        assert get_model_max_completion_tokens("ghost", default=2048) == 2048

    def test_default_value_is_respected(self, mock_catalog):
        assert get_model_max_completion_tokens("gpt-4o", default=8192) == 8192


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------

class TestResolveModel:

    def test_no_args_returns_default_model(self, mock_catalog):
        # DEFAULT_MODEL = "gpt-4o" which is in SAMPLE_CATALOG
        assert resolve_model() == "gpt-4o"

    def test_requested_model_returned(self, mock_catalog):
        assert resolve_model(requested_model="gpt-5") == "gpt-5"

    def test_requested_model_not_in_catalog_raises(self, mock_catalog):
        with pytest.raises(ValueError, match="not configured"):
            resolve_model(requested_model="unknown-model")

    def test_requested_model_not_vision_capable_raises(self, mock_catalog):
        with pytest.raises(ValueError, match="not vision-capable"):
            resolve_model(requested_model="text-only-model", require_vision=True)

    def test_prefer_model_used_when_default_skipped(self, mock_catalog, monkeypatch):
        # Patch DEFAULT_MODEL to something not in catalog so prefer_model wins
        monkeypatch.setattr(config_module, "DEFAULT_MODEL", "nonexistent")
        assert resolve_model(prefer_model="gpt-4o-mini") == "gpt-4o-mini"

    def test_prefer_model_ignored_if_not_vision_capable(self, mock_catalog, monkeypatch):
        # prefer_model has no vision, DEFAULT_MODEL "gpt-4o" does → falls through to gpt-4o
        assert resolve_model(prefer_model="text-only-model", require_vision=True) == "gpt-4o"

    def test_require_vision_skips_non_vision_models(self, mock_catalog, monkeypatch):
        # Remove DEFAULT_MODEL from catalog so it falls through to first vision model
        monkeypatch.setattr(config_module, "DEFAULT_MODEL", "nonexistent")
        result = resolve_model(require_vision=True)
        assert result in {"gpt-5", "gpt-4o", "gpt-4o-mini"}

    def test_falls_through_to_first_available_when_defaults_missing(self, mock_catalog, monkeypatch):
        monkeypatch.setattr(config_module, "DEFAULT_MODEL", "nonexistent")
        result = resolve_model()
        # Should be the first model in SAMPLE_CATALOG that isn't "nonexistent"
        assert result in {"gpt-5", "gpt-4o", "gpt-4o-mini", "text-only-model"}

    def test_no_compatible_models_raises(self, monkeypatch):
        all_text_catalog = {
            "config": {"pricing_unit": 1_000_000, "monthly_limit": 250.0},
            "models": {
                "text-a": {"input": 0.1, "output": 0.1, "supports_vision": False},
                "text-b": {"input": 0.2, "output": 0.2, "supports_vision": False},
            },
        }
        monkeypatch.setattr(config_module, "load_model_catalog", lambda: all_text_catalog)
        monkeypatch.setattr(config_module, "DEFAULT_MODEL", "nonexistent")
        with pytest.raises(ValueError, match="No vision-capable models"):
            resolve_model(require_vision=True)

    def test_requested_model_takes_precedence_over_prefer(self, mock_catalog):
        assert resolve_model(requested_model="gpt-5", prefer_model="gpt-4o-mini") == "gpt-5"


# ---------------------------------------------------------------------------
# save_model_catalog
# ---------------------------------------------------------------------------

class TestSaveModelCatalog:

    def test_saves_json_to_catalog_path(self, monkeypatch, tmp_path):
        output_file = tmp_path / "model_catalog.json"
        monkeypatch.setattr(config_module, "get_model_catalog_path", lambda: output_file)
        save_model_catalog(SAMPLE_CATALOG)
        assert output_file.exists()
        loaded = json.loads(output_file.read_text())
        assert loaded["config"]["pricing_unit"] == 1_000_000
        assert "gpt-4o" in loaded["models"]

    def test_saved_json_is_valid_and_round_trips(self, monkeypatch, tmp_path):
        output_file = tmp_path / "model_catalog.json"
        monkeypatch.setattr(config_module, "get_model_catalog_path", lambda: output_file)
        save_model_catalog(SAMPLE_CATALOG)
        round_tripped = json.loads(output_file.read_text())
        assert round_tripped == SAMPLE_CATALOG
