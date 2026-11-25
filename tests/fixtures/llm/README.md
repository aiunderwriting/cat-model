# LLM transport goldens

`request_payloads.json` pins the request each provider mechanism sends: for an OpenAI background run, the body and path LiteLLM's Responses API submits, captured from an in-memory API; for a stream (Anthropic, xAI, Google, Meta and OpenRouter), the keyword arguments of `litellm.acompletion`. `tests/test_llm_litellm_provider.py` uses it without a live provider call.
