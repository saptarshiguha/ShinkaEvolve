# Guide to Changing LLM Backends in ShinkaEvolve

This guide explains how to modify the LLM backend to use different providers or add new LLM providers to ShinkaEvolve.

---

## Overview

ShinkaEvolve uses LLMs in **four different roles**:

1. **Code Mutation LLMs**: Generate code patches to evolve programs
2. **Meta-Learning LLMs**: Generate meta-recommendations from successful programs
3. **Novelty Judgment LLMs**: Assess if programs are meaningfully different
4. **Embedding Models**: Convert code to vector representations

Each can use different models/providers. This guide shows you how to modify all of them.

---

## Quick Start: Changing Existing Providers

### For Code Mutation LLMs

**File**: Your evolution configuration (e.g., `examples/adas_aime/run_evo.py`)

**Location**: Lines 63-67

```python
evo_config = EvolutionConfig(
    # CHANGE THESE MODELS:
    llm_models=[
        "gemini-2.5-pro",                                    # Google Gemini
        "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0", # AWS Bedrock Claude
        "azure-o4-mini",                                      # Azure OpenAI
    ],
    llm_kwargs=dict(temperatures=[0.0, 0.5, 1.0], max_tokens=16384),
    ...
)
```

**Supported Providers Out-of-the-Box**:
- **OpenAI**: `"gpt-4.1"`, `"gpt-4.1-mini"`, `"gpt-4o-mini"`, `"o1-2024-12-17"`, etc.
- **Azure OpenAI**: `"azure-gpt-4.1"`, `"azure-o4-mini"` (prefix with `azure-`)
- **Anthropic**: `"claude-3-5-sonnet-20241022"`, `"claude-4-sonnet-20250514"`, etc.
- **AWS Bedrock**: `"bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"` (prefix with `bedrock/`)
- **Google Gemini**: `"gemini-2.5-pro"`, `"gemini-2.0-flash-thinking-exp-01-21"`, etc.
- **DeepSeek**: `"deepseek-chat"`, `"deepseek-reasoner"`, etc.

---

### For Meta-Learning LLMs

**File**: Your evolution configuration

**Location**: Lines 70-71

```python
evo_config = EvolutionConfig(
    ...
    # CHANGE THESE:
    meta_llm_models=["azure-gpt-4.1"],  # Model for generating insights
    meta_llm_kwargs=dict(temperatures=[0.0]),  # Lower temp for analysis
    ...
)
```

**Recommendation**: Use cheaper, faster models for meta-learning (e.g., `gpt-4.1-mini`, `claude-3-5-haiku`)

---

### For Novelty Judgment LLMs

**File**: Your evolution configuration

**Location**: Lines 76-77

```python
evo_config = EvolutionConfig(
    ...
    # CHANGE THESE:
    novelty_llm_models=["azure-gpt-4.1"],  # Model for judging novelty
    novelty_llm_kwargs=dict(temperatures=[0.0]),  # Deterministic judgment
    ...
)
```

**Recommendation**: Use fast, cheap models (novelty checks happen frequently)

---

### For Embedding Models

**File**: Your evolution configuration

**Location**: Line 72

```python
evo_config = EvolutionConfig(
    ...
    # CHANGE THIS:
    embedding_model="text-embedding-3-small",  # OpenAI embedding model
    ...
)
```

**Supported Embedding Models**:
- OpenAI: `"text-embedding-3-small"`, `"text-embedding-3-large"`
- (Other providers require custom implementation)

---

## Adding a New LLM Provider

If you want to add a completely new provider (e.g., Cohere, Together AI, local models), follow these steps:

### Step 1: Add Pricing Information

**File**: `shinka/llm/models/pricing.py`

**What to Add**:

```python
# Add your provider's pricing dictionary
YOUR_PROVIDER_MODELS = {
    "your-model-name": {
        "input_price": 1.0 / M,   # Price per 1M input tokens
        "output_price": 3.0 / M,  # Price per 1M output tokens
    },
    "your-other-model": {
        "input_price": 0.5 / M,
        "output_price": 1.5 / M,
    },
}

# If your provider has reasoning models, add them here:
REASONING_YOUR_PROVIDER_MODELS = [
    "your-reasoning-model",
]
```

**Example**:

```python
# Add at the end of pricing.py
COHERE_MODELS = {
    "command-r-plus": {
        "input_price": 3.0 / M,
        "output_price": 15.0 / M,
    },
    "command-r": {
        "input_price": 0.5 / M,
        "output_price": 1.5 / M,
    },
}
```

---

### Step 2: Create Provider Query Function

**File**: Create `shinka/llm/models/your_provider.py`

**What to Implement**:

```python
import backoff
import your_provider_sdk  # Your provider's Python SDK
from .pricing import YOUR_PROVIDER_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


def backoff_handler(details):
    """Handle backoff retries."""
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"YourProvider - Retry {details['tries']} due to error: {exc}. "
            f"Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        your_provider_sdk.APIConnectionError,
        your_provider_sdk.RateLimitError,
        # Add other exceptions your provider might raise
    ),
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_your_provider(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query your provider's model."""

    # 1. Build message history in your provider's format
    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    # 2. Call your provider's API
    if output_model is None:
        response = client.chat.completions.create(  # Adjust to your API
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            **kwargs,
        )
        content = response.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": content})
    else:
        # Handle structured output if your provider supports it
        raise NotImplementedError("Structured output not supported yet.")

    # 3. Calculate costs
    input_cost = (
        YOUR_PROVIDER_MODELS[model]["input_price"]
        * response.usage.input_tokens
    )
    output_cost = (
        YOUR_PROVIDER_MODELS[model]["output_price"]
        * response.usage.output_tokens
    )

    # 4. Return QueryResult
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",  # Add if your provider supports chain-of-thought
        model_posteriors=model_posteriors,
    )
    return result
```

---

### Step 3: Register Provider in Client Setup

**File**: `shinka/llm/client.py`

**Function**: `get_client_llm()`

**What to Add**:

```python
from .models.pricing import (
    CLAUDE_MODELS,
    OPENAI_MODELS,
    # ... existing imports
    YOUR_PROVIDER_MODELS,  # Add your import
)


def get_client_llm(model_name: str, structured_output: bool = False):
    """Get the client and model for the given model name."""

    # ... existing if/elif blocks for other providers

    # Add your provider check:
    elif model_name in YOUR_PROVIDER_MODELS.keys():
        import your_provider_sdk

        client = your_provider_sdk.Client(
            api_key=os.getenv("YOUR_PROVIDER_API_KEY"),
            # Add other initialization params
        )

        if structured_output:
            # Add instructor wrapping if supported
            client = instructor.from_your_provider(client)

    else:
        raise ValueError(f"Model {model_name} not supported.")

    return client, model_name
```

---

### Step 4: Register Provider in Query Router

**File**: `shinka/llm/query.py`

**Function**: `query()`

**What to Add**:

Import your query function:

```python
from .models import (
    query_anthropic,
    query_openai,
    query_deepseek,
    query_gemini,
    query_your_provider,  # Add this
    QueryResult,
)

from .models.pricing import (
    CLAUDE_MODELS,
    BEDROCK_MODELS,
    OPENAI_MODELS,
    DEEPSEEK_MODELS,
    GEMINI_MODELS,
    YOUR_PROVIDER_MODELS,  # Add this
)
```

Add routing logic (around line 160+):

```python
def query(
    model_name: str,
    msg: str,
    system_msg: str = "",
    msg_history: List[Dict] = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict] = None,
    **kwargs,
) -> QueryResult:
    """Route query to appropriate provider."""

    client, model = get_client_llm(model_name, output_model is not None)

    # ... existing routing

    # Add your provider:
    elif model_name in YOUR_PROVIDER_MODELS.keys():
        return query_your_provider(
            client,
            model,
            msg,
            system_msg,
            msg_history,
            output_model,
            model_posteriors,
            **kwargs,
        )

    else:
        raise ValueError(f"Model {model_name} not supported.")
```

---

### Step 5: Add Environment Variables

**File**: `.env` (in repository root)

**What to Add**:

```bash
# Your Provider API Key
YOUR_PROVIDER_API_KEY=your_api_key_here

# Any other provider-specific config
YOUR_PROVIDER_BASE_URL=https://api.yourprovider.com
```

---

### Step 6: Update Model Exports

**File**: `shinka/llm/models/__init__.py`

**What to Add**:

```python
from .anthropic import query_anthropic
from .openai import query_openai
from .deepseek import query_deepseek
from .gemini import query_gemini
from .your_provider import query_your_provider  # Add this
from .result import QueryResult

__all__ = [
    "query_anthropic",
    "query_openai",
    "query_deepseek",
    "query_gemini",
    "query_your_provider",  # Add this
    "QueryResult",
]
```

---

### Step 7: Test Your Provider

Create a test script:

```python
# test_your_provider.py
from shinka.llm import LLMClient

# Test single model
client = LLMClient(
    model_names=["your-model-name"],
    temperatures=0.7,
    max_tokens=1000,
)

response = client.query(
    msg="What is 2+2?",
    system_msg="You are a helpful assistant.",
)

print(f"Response: {response.content}")
print(f"Cost: ${response.cost:.6f}")
print(f"Tokens: {response.input_tokens} in, {response.output_tokens} out")
```

Run it:

```bash
python test_your_provider.py
```

---

## Advanced: Custom Embedding Provider

If you want to use a different embedding provider:

### Step 1: Create Embedding Client

**File**: Create `shinka/llm/embedding_your_provider.py`

```python
from typing import List, Tuple
import your_embedding_sdk


class YourProviderEmbeddingClient:
    def __init__(self, model_name: str, verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self.client = your_embedding_sdk.Client()

    def get_embedding(self, text: str) -> Tuple[List[float], float]:
        """Get embedding vector for text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model_name,
        )

        embedding = response.data[0].embedding
        cost = 0.0  # Calculate based on your provider's pricing

        return embedding, cost
```

### Step 2: Modify EmbeddingClient

**File**: `shinka/llm/embedding.py`

**Add provider detection**:

```python
class EmbeddingClient:
    def __init__(self, model_name: str, verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose

        # Add your provider detection
        if model_name.startswith("your-provider-"):
            from .embedding_your_provider import YourProviderEmbeddingClient
            self.client = YourProviderEmbeddingClient(model_name, verbose)
        elif model_name.startswith("text-embedding"):
            # Existing OpenAI logic
            self.client = openai.OpenAI()
        else:
            raise ValueError(f"Embedding model {model_name} not supported")
```

---

## Common Modifications by Use Case

### Use Case 1: Switch from OpenAI to Anthropic

**Goal**: Replace all OpenAI models with Anthropic Claude

**Changes Needed**:

```python
# Before (run_evo.py):
llm_models=["azure-gpt-4.1", "azure-o4-mini"],
meta_llm_models=["azure-gpt-4.1"],
novelty_llm_models=["azure-gpt-4.1"],
embedding_model="text-embedding-3-small",

# After:
llm_models=["claude-4-sonnet-20250514", "claude-3-5-haiku-20241022"],
meta_llm_models=["claude-3-5-haiku-20241022"],
novelty_llm_models=["claude-3-5-haiku-20241022"],
embedding_model="text-embedding-3-small",  # Keep OpenAI (no Claude embedding)
```

**Environment Variables**:

```bash
# .env
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_for_embeddings
```

---

### Use Case 2: Use Local Models (Ollama, vLLM)

**Goal**: Run evolution with local models

**Changes Needed**:

1. **Create OpenAI-compatible wrapper** (Ollama and vLLM support OpenAI API format):

```python
# In your configuration:
llm_models=["ollama-llama-3.1-70b"],  # Prefix with "ollama-"
```

2. **Modify client.py**:

```python
elif model_name.startswith("ollama-"):
    # Remove prefix for actual model name
    actual_model = model_name.replace("ollama-", "")

    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",  # Ollama API endpoint
        api_key="ollama",  # Dummy key
    )
    model_name = actual_model  # Use actual model name
```

3. **Add to pricing.py** (set to 0 for local models):

```python
OLLAMA_MODELS = {
    "llama-3.1-70b": {
        "input_price": 0.0,  # Local = free
        "output_price": 0.0,
    },
}
```

---

### Use Case 3: Use AWS Bedrock Instead of Direct Anthropic

**Goal**: Use Claude through AWS Bedrock instead of Anthropic API

**Changes Needed**:

```python
# Configuration:
llm_models=[
    "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",  # AWS region + model ID
],
```

**Environment Variables**:

```bash
# .env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION_NAME=us-east-1  # Your AWS region
```

**Already Supported**: No code changes needed! The `bedrock/` prefix is detected in `client.py:39-49`.

---

### Use Case 4: Mix Multiple Providers

**Goal**: Use best model from each provider for different roles

**Configuration**:

```python
evo_config = EvolutionConfig(
    # Code mutations: Use multiple providers
    llm_models=[
        "claude-4-sonnet-20250514",     # Anthropic - creative
        "gemini-2.5-pro",                # Google - versatile
        "azure-gpt-4.1",                 # Azure OpenAI - reliable
    ],

    # Meta-learning: Cheap and fast
    meta_llm_models=["gpt-4.1-mini"],

    # Novelty: Cheap and fast
    novelty_llm_models=["claude-3-5-haiku-20241022"],

    # Embeddings: OpenAI
    embedding_model="text-embedding-3-small",
)
```

**Environment Variables Needed**:

```bash
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
AZURE_OPENAI_API_KEY=...
AZURE_API_VERSION=...
AZURE_API_ENDPOINT=...
OPENAI_API_KEY=...  # For embeddings
```

---

## File Reference Summary

Here's every file you might need to modify:

| Task | Files to Modify | Functions/Classes |
|------|----------------|-------------------|
| **Change existing models** | Your config (e.g., `examples/adas_aime/run_evo.py`) | `EvolutionConfig` parameters |
| **Add new provider** | `shinka/llm/models/pricing.py` | Add `YOUR_PROVIDER_MODELS` dict |
| | `shinka/llm/models/your_provider.py` | Create `query_your_provider()` |
| | `shinka/llm/client.py` | Modify `get_client_llm()` |
| | `shinka/llm/query.py` | Modify `query()` routing |
| | `shinka/llm/models/__init__.py` | Add export |
| | `.env` | Add API keys |
| **Add embedding provider** | `shinka/llm/embedding_your_provider.py` | Create embedding client class |
| | `shinka/llm/embedding.py` | Modify `EmbeddingClient.__init__()` |
| **Test changes** | Create `test_your_provider.py` | Test script |

---

## Debugging Tips

### Problem: "Model not supported" error

**Solution**: Check that model name matches exactly in pricing.py

```python
# Check available models:
from shinka.llm.models.pricing import OPENAI_MODELS
print(OPENAI_MODELS.keys())
```

### Problem: API authentication fails

**Solution**: Verify environment variables are loaded

```python
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(".env")
load_dotenv(dotenv_path=env_path, override=True)
print(f"API Key: {os.getenv('YOUR_PROVIDER_API_KEY')[:10]}...")  # Print first 10 chars
```

### Problem: Cost calculation is wrong

**Solution**: Update pricing in `pricing.py`. Check provider's pricing page:
- Anthropic: https://www.anthropic.com/pricing
- OpenAI: https://openai.com/pricing
- Google: https://ai.google.dev/gemini-api/docs/pricing

### Problem: Retries happening frequently

**Solution**: Increase backoff parameters in your `query_your_provider.py`:

```python
@backoff.on_exception(
    backoff.expo,
    (...),
    max_tries=30,      # Increase from 20
    max_value=60,      # Increase max wait time
)
```

---

## Complete Example: Adding Cohere

Here's a complete example of adding Cohere as a provider:

### 1. Add to pricing.py:

```python
COHERE_MODELS = {
    "command-r-plus": {
        "input_price": 3.0 / M,
        "output_price": 15.0 / M,
    },
    "command-r": {
        "input_price": 0.5 / M,
        "output_price": 1.5 / M,
    },
}
```

### 2. Create models/cohere.py:

```python
import backoff
import cohere
from .pricing import COHERE_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


@backoff.on_exception(
    backoff.expo,
    (cohere.CohereAPIError, cohere.CohereConnectionError),
    max_tries=20,
    max_value=20,
)
def query_cohere(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Cohere model."""

    # Cohere uses preamble for system message
    response = client.chat(
        model=model,
        message=msg,
        preamble=system_msg,
        chat_history=msg_history,
        **kwargs,
    )

    content = response.text

    # Update history
    new_msg_history = msg_history + [
        {"role": "USER", "message": msg},
        {"role": "CHATBOT", "message": content},
    ]

    # Calculate costs
    input_tokens = response.meta.tokens.input_tokens
    output_tokens = response.meta.tokens.output_tokens
    input_cost = COHERE_MODELS[model]["input_price"] * input_tokens
    output_cost = COHERE_MODELS[model]["output_price"] * output_tokens

    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",
        model_posteriors=model_posteriors,
    )
```

### 3. Modify client.py:

```python
from .models.pricing import (
    # ... existing
    COHERE_MODELS,
)

def get_client_llm(model_name: str, structured_output: bool = False):
    # ... existing checks

    elif model_name in COHERE_MODELS.keys():
        import cohere
        client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

    # ... rest
```

### 4. Modify query.py:

```python
from .models import (
    # ... existing
    query_cohere,
)
from .models.pricing import (
    # ... existing
    COHERE_MODELS,
)

def query(...):
    # ... existing routing

    elif model_name in COHERE_MODELS.keys():
        return query_cohere(
            client, model, msg, system_msg, msg_history,
            output_model, model_posteriors, **kwargs
        )
```

### 5. Update .env:

```bash
COHERE_API_KEY=your_cohere_key_here
```

### 6. Use in config:

```python
llm_models=["command-r-plus", "command-r"],
```

Done! Cohere is now fully integrated.

---

## Questions?

If you encounter issues:

1. Check the existing provider implementations in `shinka/llm/models/`
2. Look at how similar providers are handled (e.g., OpenAI vs Azure OpenAI)
3. Verify your API keys are correct in `.env`
4. Test with a simple script before using in evolution

The modular design makes it easy to swap providers or add new ones without changing core evolution logic!
