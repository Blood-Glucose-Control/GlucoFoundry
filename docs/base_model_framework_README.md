# Base Model Runtime and Extension Contract

This document defines how model construction works in the repository and what
contributors must implement when adding a new model family.

## Current constructor paths

There are currently two constructor utilities in active runtime use:

1. [`create_model_and_config`](/src/models/factory.py:35)
   - used by evaluation/personalization entrypoints such as:
     - [`nocturnal_hypo_eval.py`](/src/workflows/evaluation/nocturnal_hypo_eval.py)
     - [`per_patient_finetune.py`](/src/workflows/personalization/per_patient_finetune.py)
2. [`ModelFactory.create_model`](/src/workflows/forecasting/modeling.py:124) and [`ModelFactory.load_model`](/src/workflows/forecasting/modeling.py:601)
   - used by forecasting pipeline workflows:
     - [`pipeline.py`](/src/workflows/forecasting/pipeline.py)

In plain terms: constructor behavior is not fully unified yet; both utilities are
live, and both end up instantiating the same concrete model families.

## Config validation boundary

Model YAML overrides should be validated via:

- [`load_model_config_from_yaml`](/src/workflows/forecasting/modeling.py:50)
- schema routes in [`MODEL_CONFIG_ROUTES`](/src/config/schemas/model_configs.py:830)

## ModelRegistry role

[`ModelRegistry`](/src/models/base/registry.py) is intentionally kept for:

- model-class registration via decorators,
- dependency-aware lazy lookup/discovery,
- registry contract tests.

It is not the runtime constructor used by evaluation/pipeline entrypoints.

## Required checklist for adding a model family

1. Add family code under [`src/models/`](/src/models).
2. Set `config_class = <FamilyConfig>` on the forecaster class.
3. Register the model class with [`ModelRegistry.register`](/src/models/base/registry.py:32).
4. Add schema + runtime adapter in [`src/config/schemas/model_configs.py`](/src/config/schemas/model_configs.py).
5. Wire constructor handling in currently active constructor paths.
6. Add/update tests:
   - [`test_model_config_schema_loader.py`](/tests/workflows/forecasting/test_model_config_schema_loader.py)
   - [`test_model_family_contract_suite.py`](/tests/models/test_model_family_contract_suite.py)
   - family-specific tests in [`tests/models/`](/tests/models)

## Validation commands

```bash
ruff check src/config/schemas src/workflows/forecasting src/workflows/evaluation src/models/<family>
pytest -q tests/workflows/forecasting/test_model_config_schema_loader.py
pytest -q tests/models/test_model_family_contract_suite.py
```
