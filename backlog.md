
# Backlog - Agents7

## Improvements

- Add proper batch support by changing the prompts.
- I think I should create different Detector types: for classification,
  or for inference.
- Add possibility to add prompt parameters in all agents.
  I have to think how to do it properly.
- Check `pydantic_aio` library, it seems interesting.


## Fixes

- Interface for the model `gpt-oss:20b` in `ollama` is not working.
  I have to check if the library `ollama` is updated, or if the model
  has changed the interface. It returns empty responses.
