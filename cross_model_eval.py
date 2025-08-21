
"""
LLM Evaluation Framework for Academic Research (2025)
----------------------------------------------------

This script provides a framework for systematically evaluating a representative selection
of leading Large Language Models (LLMs) across different families/providers.

MODEL FAMILY SELECTION RATIONALE:
---------------------------------

1. OpenAI (ChatGPT Family)
   - Industry pioneer and current market leader
   - Consistently top performer on most benchmarks
   - Represents the commercial state-of-the-art baseline

2. Anthropic (Claude Family)
   - Known for strong reasoning capabilities and safety features
   - Different training methodology ("Constitutional AI")
   - Distinguished by long context windows and strong instruction following

3. Google (Gemini Family)
   - Developed with a focus on multimodal capabilities
   - Strong performer on mathematical and scientific reasoning
   - Represents a different training philosophy from OpenAI/Anthropic

4. Mistral AI
   - Leading European AI lab with rapidly improving models
   - Known for efficient models that perform well at smaller scales
   - Mixture-of-experts architecture provides efficiency advantages

5. Qwen (Alibaba)
   - Leading model family from Asia
   - Strong multilingual capabilities, especially for Chinese
   - Represents different training data distribution and cultural context
   - Provides geographic diversity in model comparison

6. DeepSeek
   - Known for specialized variants (coding, reasoning)
   - Strong performer on technical and academic benchmarks
   - Represents emerging players in the open model space

7. Meta (Llama Family)
   - Most widely adopted open weight models
   - Strong research focus and community adoption
   - Provides important open-source baseline for comparison

This selection achieves a balanced representation across:
- Commercial vs. open-source models
- US vs.Chinese vs. European development approaches
- Different technical architectures and training methodologies
- Established players vs. emerging competitors

Together, these model families provide a comprehensive snapshot of the
current state-of-the-art in general-purpose language models.
"""

import os
import time
import pandas as pd
import json
from datetime import datetime
from litellm import completion

# Load API keys from .env file
from dotenv import load_dotenv
load_dotenv()

# API keys will be automatically loaded from .env into environment variables
# Make sure .env file contains:
# OPENAI_API_KEY=your-openai-key
# ANTHROPIC_API_KEY=your-anthropic-key 
# GOOGLE_API_KEY=your-google-key
# MISTRAL_API_KEY=your-mistral-key

# Organize models by family with variants for each
# For each family, we start by running on a full size model
# If cost and time allows, we will also try lite models, and potentially a powerful reasoning model from each provider where possible
# Model names correpsond to the LiteLLM documentation, e.g. https://docs.litellm.ai/docs/providers/openai
model_families = {
    "OpenAI": [
        # Large model
        "gpt-4.1",
        # Lite model
        #"gpt-4.1-mini",
        # Most powerful reasoning model
        #"o3"
    ],
    "Anthropic": [
        # Large model
        "claude-3-5-sonnet",
        # Lite model
        #"claude-3-haiku",
        # Most powerful reasoning model
        #"anthropic/claude-3.7-sonnet"
    ],
    "Google": [
        # Large model
        "gemini/gemini-2.0-flash",
        # Lite model
        #"gemini/gemini-2.0-flash-lite"
        # Most powerful reasoning model
        #"gemini/gemini-2.5-pro"
    ],
    # "Mistral": [
    #     # Large model
    #     "mistral/mistral-medium-2505"
    #     # Lite model
    #     #"mistral/mistral-small-2503"
    #     # Most powerful reasoning model
    #     #"mistral/mistral-large-2411",
    # ],
    # "Qwen": [
    #     # Large model
    #     "ollama/qwen-72b",
    #     # Liter model
    #     "ollama/qwen-7b"
    #     # No Qwen reasoning models
    # ],
    # "DeepSeek": [
    #     # Most powerful reasoning model
    #     "ollama/deepseek-v3",
    # ],
    # "Meta": [
    #     # Larger model
    #     "ollama/llama-4-maverick",
    #     # Lite model
    #     "ollama/llama-4-scout",
    #     # Most powerful reasoning model
    #     # ???
    # ]
}

# Now use litellm to query one of the models
response = completion(
    model=model_families["OpenAI"][0],
    prompt="Hello, how are you?"
)

print(response)

