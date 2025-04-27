from local_embed import LocalEmbedding
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from fetch_news import get_fmp_news, get_forex_news
import json

def retrival(NEWS_DATA, question: str):
    retrieve_start_time = time.time()
    # set up the embedding model
    embed_model = LocalEmbedding(model_name=embed_model_name, device=device)

    # Initialize tokenizer for chunking
    tokenizer = AutoTokenizer.from_pretrained(embed_model_name)

    # Split documents into chunks
    chunks = []
    for date, content in NEWS_DATA:
        text = f"Date: {date}\nContent: {content}"
        step = chunk_size - chunk_overlap if chunk_size > chunk_overlap else chunk_size
        token_ids = tokenizer.encode(text, add_special_tokens=False,max_length=chunk_size, truncation=True)
        for i in range(0, len(token_ids), step):
            chunk_ids = token_ids[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)


    # Compute embeddings for all chunks
    chunk_embeddings = embed_model.model.encode(
        chunks,
        convert_to_tensor=True
    ).to(device)

    # Compute embedding for the query
    query_embedding = embed_model.model.encode(
        question,
        convert_to_tensor=True
    ).to(device)

    # Cosine similarity between query and each chunk
    similarity_scores = F.cosine_similarity(
        chunk_embeddings,
        query_embedding.unsqueeze(0),
        dim=1
    )

    # Retrieve top-k chunks
    topk_scores, topk_idx = torch.topk(similarity_scores, k=similarity_top_k)
    topk_idx = topk_idx.tolist()


    # Concatenate retrieved chunks into context
    retrieved_chunks = [chunks[i] for i in topk_idx]

    # Build the RAG prompt
    context = "\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n### Answer:"
    retrieval_time = time.time() - retrieve_start_time
    print(f"Retrieval Time: {retrieval_time:.2f} seconds")
    return prompt, retrieval_time


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
        print(f"Average time taken for similarity_top_k = {k}: {total_time / len(lines):.2f} seconds")

    for k, time_taken in k_time.items():
        print(f"Average time taken for similarity_top_k = {k}: {time_taken:.2f} seconds")





