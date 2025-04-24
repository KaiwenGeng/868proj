from fetch_news import get_fmp_news, get_forex_news
from pydantic import ConfigDict
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from local_embed import LocalEmbedding

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Any, Iterator, Dict
import torch
import time
class LocalLLM(CustomLLM):
    context_window: int = 8192
    num_output: int = 2048
    model_name: str = "Meta-Llama-3-8B-Instruct"
    model_config = ConfigDict(extra="allow")

    def __init__(self, model_path: str, device="cuda:0"):
        super().__init__()

        # bnb_config = BitsAndBytesConfig(
        #     load_in_8bit=True,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=True,
        #     bnb_8bit_compute_dtype=torch.float16,
        # )

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # nf4、fp4 ...
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.float16,
        )

        # quantization_config = (BitsAndBytesConfig
        #                        (load_in_8bit=True,
        #                         bnb_8bit_compute_dtype=torch.float16,))

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     quantization_config=bnb_config,
        #     device_map={"": device},
        #     torch_dtype=torch.float16,
        # )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map={"": device},
            low_cpu_mem_usage=True,
            # No torch_dtype=torch.float16 otherwise will raise some error
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
        start_time = time.time()

        # create documents from the news data
        documents = [
            Document(text=f"Date: {date}\nContent: {content}")
            for date, content in NEWS_DATA
        ]

        # set up the embedding model
        embed_model = LocalEmbedding(model_name=self.embed_model_name, device=self.device)
        Settings.embed_model = embed_model

        # tokenize the documents
        splitter = TokenTextSplitter(self.chunk_size, self.chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)

        # build the index
        index = VectorStoreIndex(nodes)

        # retrieval time
        retrieval_start = time.time()
        retriever = index.as_retriever(similarity_top_k=self.similarity_top_k)
        retrieved_nodes = retriever.retrieve(question)
        retrieval_end = time.time()
        print(f"Retrieval Time: {retrieval_end - retrieval_start:.2f} seconds")

        # concatenate the content of the retrieved nodes
        context = "\n".join([node.get_content() for node in retrieved_nodes])
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n### Answer:"

        # inference time
        inference_start = time.time()
        response = self.llm.complete(prompt, max_new_tokens=self.max_new_tokens)
        inference_end = time.time()
        print(f"LLM Inference Time: {inference_end - inference_start:.2f} seconds")
        print(f"Total Time: {inference_end - start_time:.2f} seconds")

        return response


class decision_maker:
    def __init__(self, llm: LocalLLM, max_new_tokens=2048):
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    def make_decision(self, prompt: str) -> str:
        start_time = time.time()
        response = self.llm.complete(prompt, max_new_tokens=self.max_new_tokens)
        end_time = time.time()
        print(f"Decision Making Time: {end_time - start_time:.2f} seconds")
        return response