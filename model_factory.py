import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


def get_model(provider: Optional[str] = None) -> BaseChatModel:
    """Instantiates a provider-agnostic BaseChatModel instance with greedy decoding."""
    selected_provider = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()

    if selected_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.0)

    elif selected_provider in ["gemini", "google"]:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set.")
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0)

    else:
        raise ValueError(
            f"Unsupported API provider '{selected_provider}'. Choose 'openai' or 'gemini' "
            f"('claude-code' is handled in main.py and does not use an API model)."
        )
