# Model provider examples

The examples in this directory show how to route models through adapter layers such as LiteLLM and
any-llm, and through locally installed CLI runtimes. The default adapter examples use OpenRouter so
you only need one API key:

```bash
export OPENROUTER_API_KEY="..."
```

Run one of the adapter examples:

```bash
uv run examples/model_providers/any_llm_provider.py
uv run examples/model_providers/any_llm_auto.py
uv run examples/model_providers/litellm_provider.py
uv run examples/model_providers/litellm_auto.py
```

Direct-model examples let you override the target model:

```bash
uv run examples/model_providers/any_llm_provider.py --model openrouter/openai/gpt-5.4-mini
uv run examples/model_providers/litellm_provider.py --model openrouter/openai/gpt-5.4-mini
```

For local CLI runtimes, make sure the target CLI is already installed and authenticated, then run:

```bash
uv run examples/model_providers/cli_provider.py --model codex
uv run examples/model_providers/cli_provider.py --model codex:gpt-5.4/high --transport acp
uv run examples/model_providers/cli_provider.py --model gemini:gemini-2.5-pro --execution-mode sdk_controlled
uv run examples/model_providers/cli_provider.py --model copilot:gpt-4.1
uv run examples/model_providers/cli_provider.py --model gemini:gemini-2.5-flash --transport acp
```
