
# Backlog - Agents7

## Improvements

- Add possibility to parameterize parts of the prompt 
  in all agents, e.g., current date, custom values, etc. 
  I have to think how to do it properly.
- Check `pydantic_aio` library, it seems interesting.


## Fixes

- [Fixed in v0.3.3] Model `gpt-oss:20b` was returning empty responses with
  structured output. Added fallback and JSON parsing fixes.
