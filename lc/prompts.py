"""The prompt object the provider adapters consume.

Every model sees byte-identical text: the system line below, then the item's context
(`cacheable`, shared by all queries of an item), then the query (`tail`). Adapters decide
how to transmit it; they never change the wording.
"""
from __future__ import annotations

from dataclasses import dataclass

SYSTEM = ("Answer the question using only the information provided. "
          "Give your final answer as a single probability between 0 and 1.")


@dataclass(frozen=True)
class Prompt:
    system: str
    cacheable: str
    tail: str

    def as_text(self) -> str:
        return self.cacheable + self.tail


def prompt_for_row(row: dict) -> Prompt:
    return Prompt(system=SYSTEM, cacheable=row["prefix"], tail=row["question"])
