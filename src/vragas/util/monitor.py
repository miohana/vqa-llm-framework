import threading
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult


class LLMUsageMonitor(BaseCallbackHandler):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def update(self, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += completion_tokens + prompt_tokens

    def __str__(self) -> str:
        return "prompt_tokens={0}, completion_tokens={1}, total_tokens={2}".format(
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens
        )

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        try:
            message = response.generations[0][0].message
        except (IndexError, AttributeError):
            message = None

        if isinstance(message, AIMessage):
            usage = message.usage_metadata or {}
        else:
            usage = {}

        self.update(
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0)
        )
