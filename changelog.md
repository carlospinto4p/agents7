
## Changelog - Agents7

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
