"""Real model setup (OpenAI by default). Keys come from the environment / .env only."""
from __future__ import annotations

import os


def get_real_model():
    from dotenv import load_dotenv

    load_dotenv()
    name = os.getenv("MODEL_NAME") or "gpt-4o-mini"
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=name, temperature=0), {"provider": "openai", "model": name}


def get_model(fake: bool = False):
    if fake:
        from .fake_model import FAKE_INFO, FakeRuleModel

        return FakeRuleModel(), FAKE_INFO
    return get_real_model()
