
## Changelog - Agents7

### v0.3.4 - 25th January 2026

- Module `agents`:
  - `Labeler`: Refactored JSON parsing helpers for cleaner code:
    - Made `_fix_malformed_json` a static method.
    - Extracted `_get_enum_mappings`, `_apply_enum_mappings` methods.
    - Moved imports to module level.


### v0.3.3 - 25th January 2026

- Module `llm`:
  - `OllamaProvider`: Added fallback for models that don't support
    structured output format (retries without format parameter).
- Module `agents`:
  - `Labeler`: Added JSON parsing fixes for common LLM output issues:
    - Handles markdown code blocks (```json...```)
    - Handles double-brace malformation ({"{"...)
    - Converts enum names to values when models return names instead
- Tests:
  - Added `test_multi_model.py` with 9 tests for multi-model support.


### v0.3.2 - 25th January 2026

- Examples:
  - Fixed YAML syntax in `demo_temperature_detector.yml` (typo in keys).
  - Improved temperature detector prompt to reason conceptually instead of
    matching keywords, with better indirect indicator handling.
  - Improved `detector.py` with human-readable output format.
- `.claude/`:
  - Added lesson: always read `rules/committing.md` before commits.


### v0.3.1 - 25th January 2026

- Documentation:
  - Updated `README.md` with current architecture (Classifier, Regressor,
    ClassificationProperty, Property[T]), examples using batch processing,
    and custom model definitions.
- Examples:
  - Updated `classifier.py` and `labeler.py` to use `act_in_batch` for
    efficient batch processing instead of looping with `act()`.
  - Improved `detector.py` with human-readable output format.
- `.claude/`:
  - Added lesson learned: prefer batch methods for list inputs.
  - Added lesson learned: include `.claude/` changes in changelog.
  - Added lesson learned: keep README aligned with changes.
  - Added `rm *.tmp*` permission to settings.json.
  - Updated workflow: auto-push committed changes to origin.


### v0.3.0 - 25th January 2026

- Module `models`:
  - Added `ClassificationProperty` base class for classification tasks.
    - Supports `category` and `reason` StrEnum fields.
    - Validates category-reason relationships via `CATEGORY_REASONS` mapping.
    - Auto-adds `NOT_APPLICABLE` to both enums.
  - Added `Sentiment` demo model with category-reason constraints.
- Module `agents`:
  - Added `Classifier` agent for categorical classification tasks.
    - Requires `ClassificationProperty` output model.
    - Includes `possible_categories` and `category_reasons_constraints` properties.
    - Prompts include category-reason mapping for LLM guidance.
  - Added `Regressor` agent for numeric value extraction tasks.
    - Uses existing `Property[T]` output model.
    - Specialized prompts for value extraction.
- Examples:
  - Added `classifier.py` example with sentiment analysis.
  - Added `demo_sentiment_classifier.yml` configuration.
- Tests:
  - Added `test_classifier.py` with 14 tests.
  - Added `test_regressor.py` with 11 tests.


### v0.2.0 - 25th January 2026

- Module `exceptions`:
  - Added `PropertyConfigurationError` for Property/Detector configuration errors.
- Module `models`:
  - Class `Property`:
    - Automatically adds `NOT_APPLICABLE` member to reason enums if not present.
    - Uses `__init_subclass__` to extend reason enum before Pydantic processes it.
- Module `agents`:
  - Class `Detector`:
    - Now uses `PropertyConfigurationError` instead of `TypeError`.
    - Removed explicit `NOT_APPLICABLE` validation (now automatic).
- Tests:
  - Added tests for automatic `NOT_APPLICABLE` addition.
  - Added test for `PropertyConfigurationError`.


### v0.1.0 - 25th January 2026

- Module `models`:
  - Class `Property`:
    - Now a generic class `Property[T]` for type-safe value extraction.
    - Improved documentation with usage examples.
- Module `agents`:
  - Class `Detector`:
    - Now validates that the reason enum has a `NOT_APPLICABLE` member.
    - Raises `ValueError` if the reason enum lacks this fallback.
- Module `models.demo`:
  - Class `Temperature`: Updated to use `Property[float]`.
- Tests:
  - Added test for `NOT_APPLICABLE` enforcement in Detector.
  - Updated COMFORTABLE test case for batch consistency.


### v0.0.19 - 25th January 2026

- Examples:
  - Labeler example improved with entity extraction use case.
  - Added `PersonInfo` model for clear person data extraction demo.
  - Updated `demo_labeler.yml` with extraction rules and examples.
  - Rewritten `labeler.py` with diverse test cases.
- Tests:
  - Replaced ambiguous COMFORTABLE temperature test case with explicit statement.


### v0.0.18 - 25th January 2026

- Module `agents`:
  - Class `Labeler`:
    - Improved `prompt_for_batch` to align with single mode prompt structure.
    - Added clear instruction for independent batch processing.
- Tests:
  - Replaced ambiguous test cases with clearer temperature statements.
  - Batch tests now verify LLM-dependent content successfully.


### v0.0.17 - 25th January 2026

- Module `agents`:
  - Class `Labeler`:
    - Fixed chunking bug in `act_in_batch` and `act_in_batch_async`:
      slice was `(i+1)+n` instead of `(i+1)*n`.
    - Methods `act_in_batch` and `act_in_batch_async` now return one
      `AgentResponse` per input item (flattened), instead of one per chunk.


### v0.0.16 - 25th January 2026

- Documentation:
  - `README.md`: Added comprehensive documentation with architecture overview,
    quick start examples, and development instructions.
  - `.claude/CLAUDE.md`: Added Claude Code guidelines with workflow,
    versioning, commit conventions, and lessons learned sections.
  - `.claude/settings.json`: Added permission settings for git, python,
    and bash commands.


### v0.0.15 - 5th October 2025

- Module `agents`:
  - Class `Labeler`:
    - Added the properties:
      - `prompt_for_batch`.
      - `output_model_list`.
    - Methods `act_in_batch` and `act_in_batch_async`:
      - Now they use the `prompt_for_batch` property.
      - Added the parameter `requests_per_prompt`.
    - Method `act_in_batch_async`:
      - Renamed the parameter `print_tqdm` to `with_tqdm`.
    - Added the methods:
      - `_act_in_batch_async_request`.
      - `_act_in_batch_request`.
  - Class `Detector`:
    - Added the properties:
      - `prompt_for_batch`.


### v0.0.14 - 28th September 2025

- Module `agents`:
  - Class `BaseAgent`: methods `act` response type changed to
    `models.AgentResponse`.
- Module `models`:
  - Submodule `base`: Added the models:
    - `AgentResponse`.
    - `LLMMetrics`.
  - Models also added to `__init__.py`.


### v0.0.13 - 28th September 2025

- Module `agents`:
  - Classes `Labeler` and `Detector`: fixed a bug in the method 
    `act_in_batch`: was returning a list with the output models
    instead of a list of the full responses.
- Added tests for the `Detector` class.


### v0.0.12 - 14th September 2025

- Module `agents`:
  - Class `Detector`: removed the "Reasons" part of the `prompt`.
  - Function `utils.model_class_to_string`:
    - Fix a bad rendering of the default values of the fields when 
      they are not defined.
    - Improved strings for descriptions.
    - Added a "Possible values" section for Enum fields.


### v0.0.11 - 14th September 2025

- Module `agents`:
  - Class `Labeler`:
    - Added the parameter `examples`.
    - Added the property `prompt`.
    - Added the `classmethod` `_load_yaml_settings`.
    - Added the `__posrt_init__` method.
    - With these changes, the prompt will better include 
      instructions, examples, and the output schema description.
  - Class `Detector`:
    - Added the properties `prompt` and `possible_reasons`.
    - Fix in the method `from_yaml`, it wasn't returning a 
      `Detector` instance.
  - Added the scrip `utils` with functions:
    - `get_examples_str`.
    - `import_from_path`.
    - `model_class_to_string`.
- `detector` example improved.


### v0.0.10 - 31st August 2025

- Fix in the function `llm.ollama.parse_ollama_response`,
  was not properly parsing the `thinking` response from different
  models.
- Dependency on `ollama` updated to `0.5.3`.


### v0.0.9 - 24th August 2025

- Module `agents`:
  - Added `detector.Detector` into `__init__`.
  - Class `labeler.Labeler`:
    - Added async methods `act_async` and `act_in_batch_async`.
    - Added the method `_parse_model_response`.


### v0.0.8 - 13th August 2025

- Added the class `agents.detector.Detector`, which is a simple 
  `Labeler` with a defined output model.
- Module `models`
  - Submodule `base`:
    - Just the class `Property` remains, with simplified logic. 
      No need for `PropertyReason` anymore.
  - Submodule `demo`:
    - Added the classes `Temperature` and `TemperaturePropertyReason`.
- Added a folder for `examples`, showcasing the usage of a `labeler` 
  and a `detector` agents.


### v0.0.7 - 3rd August 2025

- Module `models.base`:
  - Class `PropertyReason`: It is not an `Enum` anymore.
  - Class `Property`: logic of the method `__init_subclass__` improved.
- Function `agents.labeler.Labeler.act_in_batch` receives the
  parameter `iter_func`, which defaults to `tqdm.tqdm`.


### v0.0.6 - 3rd August 2025

- Class `models.PropertyReason`: added a field `NOT_APPLICABLE`.
  All properties should have a reason, and this is the default
  when no property-reason is applicable.


### v0.0.5 - 27th July 2025

- Added the module `models.base` with the classes `Property` and
  `PropertyReason`. Added to `models.__init__.py`.


### v0.0.4 - 20th July 2025

- Module `agents`:
  - Class `base.BaseAgent`:
    - Added the parameter `version`.
    - Added the class method `from_yaml`.
    - It is now a `frozen` dataclass.
  - Class `labeler.Labeler`:
    - Implemented the class method `from_yaml`.
    - It is now a `frozen` dataclass.
- Added the module `models` with just `demo.DemoModel`.
- Added a dependency to `pyyaml`.


### v0.0.3 - 13th July 2025

- Added the module `agents`, with:
  - The type `Agent` and the class `Labeler` in `__init__`.
  - Added the submodule `base` with the class `BaseAgent`.
  - Added the submodule `labeler` with the class `Labeler`.
- Added the module `logs`.
- Fix in `llm.ollama.parse_ollama_chat_response`.
- Added a dependency to `pydantic` and `tqdm`.


### v0.0.2 - 6th July 2025

- Module `llm`:
  - Submodule `ollama`:
    - Added methods `chat` and `chat_async` to the `OllamaProvider` class.
    - Better parsing of the response for the `generate` method.
    - Improved the function `parse_ollama_response`. 
    - Added the function `parse_ollama_chat_response`.
  - Module `base`:
    - Added the improvements in `ollama` to the `LLMEngine` and 
      `LLMProvider` classes. Doc updated.
    - Added the attribute `model_id` to both classes.
      It appears in the output of the `get_model_info` method.


### v0.0.1 - 29th June 2025

- Initial release.
- Added the module `llm`:
  - Implemented the interface using `LLMEngine` and `LLMProvider`.
  - Implemented `ollama` as a LLM provider.
