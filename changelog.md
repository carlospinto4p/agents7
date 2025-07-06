
## Changelog - Agents7

### v0.0.2 - 6th July 2025

- MNodule `llm`:
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
