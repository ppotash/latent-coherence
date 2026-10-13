"""Provider adapters.

Each adapter normalises one vendor's API to the LLMClient protocol in base.py.
Adapters handle transport and usage accounting only -- never prompt wording,
which lives in lc.prompts so that all models see identical text.
"""
