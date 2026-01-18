from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


class LLM(ABC):
    @abstractmethod
    def query(self, message: List[Dict]) -> str:
        pass


class LocalQwen(LLM):
    def __init__(self, host: str, temperature=0, useThink=False) -> None:
        super().__init__()
        self.host = host
        self.llm = ChatOpenAI(
            base_url=host,
            temperature=temperature,
            api_key=SecretStr("noneed"),
            model="glm-4-9b",
        )
        self.useThink = False

    def query(self, message: List[Dict]) -> str:
        result = self.llm.invoke(message).content
        assert type(result) == str
        if not self.useThink:
            result = result.split("</think>")[-1]
        return result
