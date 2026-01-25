
# Backlog - Agents7

## Improvements

- Add possibility to parameterize parts of the prompt 
  in all agents, e.g., current date, custom values, etc. 
  I have to think how to do it properly.
- Check `pydantic_aio` library, it seems interesting.


## Fixes

- Interface for the model `gpt-oss:20b` in `ollama` is not working.
  I have to check if the library `ollama` is updated, or if the model
  has changed the interface. It returns empty responses.
