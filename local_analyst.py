from fetch_news import get_fmp_news, get_forex_news
from pydantic import ConfigDict
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from local_embed import LocalEmbedding

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Iterator, Dict
import torch


class LocalLLM(CustomLLM):
    context_window: int = 8192
    num_output: int = 2048
    model_name: str = "Meta-Llama-3-8B-Instruct"
    model_config = ConfigDict(extra="allow")
    tokenizer: Any = None
    model: Any = None

    def __init__(self, model_path: str, torch_dtype=torch.float16, device="cuda"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        self.model.eval()

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def _complete(self, prompt: str, **kwargs: Any) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.num_output),
                do_sample=kwargs.get("do_sample", True),
                temperature=kwargs.get("temperature", 0.6),
                top_p=kwargs.get("top_p", 0.9),
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = text.split("### Answer:")[-1].strip()
        return CompletionResponse(text=answer)

    def _stream_complete(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield self._complete(prompt, **kwargs)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return self._complete(prompt, **kwargs)

    def stream_complete(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        return self._stream_complete(prompt, **kwargs)


class News_Analyst:
    def __init__(
        self,
        embed_model_name: str,
        llm: LocalLLM,
        device: str,
        chunk_size: int,
        chunk_overlap: int,
        similarity_top_k: int = 10,
        max_new_tokens: int = 2048,
        verbose: bool = False,
    ):
        self.embed_model_name = embed_model_name
        self.llm = llm
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

    def analyze(self, NEWS_DATA, question: str) -> str:
        documents = [
            Document(text=f"Date: {date}\nContent: {content}")
            for date, content in NEWS_DATA
        ]
        embed_model = LocalEmbedding(model_name=self.embed_model_name, device=self.device)
        Settings.embed_model = embed_model

        splitter = TokenTextSplitter(self.chunk_size, self.chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)

        query_engine = index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
            llm=self.llm
        )
        response = query_engine.query(question)
        return response.response


class decision_maker:
    def __init__(self, llm: LocalLLM, max_new_tokens=2048):
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    def make_decision(self, prompt: str) -> str:
        return self.llm.complete(prompt, max_new_tokens=self.max_new_tokens)