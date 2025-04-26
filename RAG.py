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
import time
from fetch_news import get_fmp_news, get_forex_news
import json



def retrival(NEWS_DATA, question: str):
    retrieve_start_time = time.time()
    general_documents = [
        Document(text=f"Date: {date}\nContent: {content}")
        for date, content in NEWS_DATA
    ]

    # set up the embedding model
    embed_model = LocalEmbedding(model_name=embed_model_name, device=device)
    Settings.embed_model = embed_model

    # tokenize the documents
    splitter = TokenTextSplitter(chunk_size, chunk_overlap)  # splitter the documents
    nodes = splitter.get_nodes_from_documents(general_documents)  # split the documents into chunks (nodes)

    # build the index
    index = VectorStoreIndex(nodes)

    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    retrieved_nodes = retriever.retrieve(question)

    # concatenate the content of the retrieved nodes
    context = "\n".join([node.get_content() for node in retrieved_nodes])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n### Answer:"
    retrieve_end_time = time.time()
    retrieval_time = retrieve_end_time - retrieve_start_time
    # print(f"Prompt:\n{prompt}")
    print(f"Retrieval Time: {retrieval_time:.2f} seconds")
    return prompt,retrieval_time


if __name__ == "__main__":
    similarity_top_k_values = [1, 2, 4, 5, 8, 10, 12, 15]
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    device = "cuda:0"
    chunk_size = 512
    chunk_overlap = 0
    similarity_top_k = 10
    api_key = "7jLb6gpseT5MzWLfY7S2K1drPwLUWFQ5"
    FOREX_OF_INTEREST = "EURUSD"
    # event = input("Enter the event description: ")
    # event = "A sovereign wealth fund announces plans to shift 10% of its portfolio from USD to Southeast Asian bonds."
    with open("events.json", "r") as f:
        lines = f.readlines()
    k_time = {}
    for k in similarity_top_k_values:
        print(f"\nRunning for similarity_top_k = {k}")
        total_time = 0
        for line in lines:
            event = json.loads(line.strip())["event"]
            question1 = f"Imagine you are an economist doing real-time event based analysis. Here is an event: {event}. What potential impact would this event have on {FOREX_OF_INTEREST} exchange rate from a macroeconomic perspective? Please provide a detailed analysis."
            question2 = f"Imagine you are a forex trader doing real-time event based trading. Here is an event: {event}. What potential impact would this event have on {FOREX_OF_INTEREST} exchange rate from a trading perspective? Please provide a detailed analysis."
            general_news = get_fmp_news(api_key)
            forex_news = get_forex_news(api_key, FOREX_OF_INTEREST)
            _, general_retrival_time = retrival(general_news, question1)
            _, forex_retrival_time = retrival(forex_news, question2)
            total_time += general_retrival_time + forex_retrival_time
        k_time[k] = total_time / len(lines)

    for k, time_taken in k_time.items():
        print(f"Average time taken for similarity_top_k = {k}: {time_taken:.2f} seconds")










