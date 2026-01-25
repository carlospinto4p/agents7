
# Backlog - Agents7

## Improvements

- [Doing] Add proper batch support by changing the prompts.
  - Added functionality to get a list of models as output in the 
    batch methods.
  - [TODO]
  - Add better output models for the batch methods.
  - Add tests!
- I think I should create different Detector types: one for classification
  and other for inference.
- Add possibility to add prompt parameters in all agents.
  I have to think how to do it properly.
- Check `pydantic_aio` library, it seems interesting.


## Fixes

- Interface for the model `gpt-oss:20b` in `ollama` is not working.
  I have to check if the library `ollama` is updated, or if the model
  has changed the interface. It returns empty responses.
