from transformers import pipeline
from langchain.llms.base import LLM
from pydantic import Extra, Field
from typing import Optional, List, Any

class LocalLLM(LLM):
    pipeline: Any = Field(default=None)

    class Config:
        extra = Extra.allow

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

    @property
    def _llm_type(self) -> str:
        return "flan_t5"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.pipeline(prompt[:1500], max_new_tokens=256)[0]["generated_text"]
