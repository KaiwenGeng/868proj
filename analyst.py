from fetch_news import get_fmp_news, get_forex_news

from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Use the correct import for HuggingFaceInferenceAPI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from huggingface_hub import login
from llama_index.core.node_parser import TokenTextSplitter

class News_Analyst:
    def __init__(self, embed_model_name, llm_model_name, device, chunk_size, chunk_overlap,
                 similarity_top_k=10, context_window=8192, max_new_tokens=2048, 
                 HF_API_TOKEN="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", verbose=True):
        self.embed_model_name = embed_model_name
        self.llm_model_name = llm_model_name
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.context_window = context_window
        self.max_new_tokens = max_new_tokens
        self.HF_API_TOKEN = HF_API_TOKEN
        self.verbose = verbose

    def analyze(self,NEWS_DATA, question):
        documents = [
            Document(text=f"Date: {date}\nContent: {content}")
            for date, content in NEWS_DATA
        ]
        # Set up the embedding model
        embed_model = HuggingFaceEmbedding(
            model_name=self.embed_model_name, device=self.device
        )
        
        # Log in to Hugging Face
        login(token=self.HF_API_TOKEN)
        
        # Create LLM instance with the correct class
        llm = HuggingFaceInferenceAPI(
            model_name=self.llm_model_name,
            token=self.HF_API_TOKEN,
            # context_window=self.context_window,
            # max_new_tokens=self.max_new_tokens,
        )
        
        # Configure settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        text_splitter = TokenTextSplitter(self.chunk_size, self.chunk_overlap)

        nodes = text_splitter.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)
        query_engine = index.as_query_engine(similarity_top_k=self.similarity_top_k)
        
        # Execute query
        response = query_engine.query(question)
        
        # Print details if verbose is True
        if self.verbose:
            print(f"Question: {question}")
            print(f"Response: {response}")
            print("Retrieved Documents:")
            for node in response.source_nodes:
                print(node.node.get_content())
                print("-" * 80)
                
        return response

class decision_maker:
    def __init__(self, llm_model_name, context_window=8192, max_new_tokens=2048, 
                 HF_API_TOKEN="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", verbose=True):
        self.llm = HuggingFaceInferenceAPI(
            model_name=llm_model_name,
            token=HF_API_TOKEN,
            # context_window=context_window,
            # max_new_tokens=max_new_tokens,
        )
    def make_decision(self, prompt):
        # just get the response from the LLM
        response = self.llm.complete(prompt)
        return response.text
        
        
