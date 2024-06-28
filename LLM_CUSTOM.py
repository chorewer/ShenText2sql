from typing import Any, List, Mapping, Optional
from dashscope import Generation
import dashscope
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY
class CustomLLM(LLM):
    n: int
    @property
    def _llm_type(self) -> str:
        return "qwen2-7b"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]   
        response = Generation.call(
            model="qwen-plus",
            messages=messages,
            seed=1234,
            result_format='message',
            max_tokens=1500,
            top_p=0.3,
            temperature=0.4,
            repetition_penalty=1,
        )
        return response.output.choices[0]['message']['content']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
    
if __name__ == "__main__":
    llm = CustomLLM(n=1)