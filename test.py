from fetch_news import get_fmp_news, get_forex_news
api_key = "7jLb6gpseT5MzWLfY7S2K1drPwLUWFQ5"
NEWS_DATA = get_fmp_news(api_key) + get_forex_news(api_key, "EURUSD")

from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.core import Settings

from huggingface_hub import login

documents = [
    Document(text=f"Date: {date}\nContent: {content}")
    for date, content in NEWS_DATA
]
print("documents filled")
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"
)

HF_API_TOKEN = "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"
login(token=HF_API_TOKEN)
llm = HuggingFaceInferenceAPI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
    token=HF_API_TOKEN,
    context_window=8192, 
    max_new_tokens=512,   
)
Settings.llm = llm
Settings.embed_model = embed_model
index = VectorStoreIndex.from_documents(documents)
print("index created")
query_engine = index.as_query_engine(similarity_top_k=10)
print("query engine created")
question = "Based on the provided documents and your knowledge, make some predictions about EUR/USD forex rates."
response = query_engine.query(question)

print("Retrieved Documents:")
for node in response.source_nodes:
    print(node.node.get_content())
    print("-" * 80)

print("\nQ:", question)
print("A:", response)