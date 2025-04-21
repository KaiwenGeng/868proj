from typing import Any
from llama_index.core.base.embeddings.base import BaseEmbedding
from sentence_transformers import SentenceTransformer

class LocalEmbedding(BaseEmbedding):
    model: Any = None

    def __init__(self, model_name, device="cpu"):
        super().__init__()
        self.model = SentenceTransformer(model_name, device=device)

    def _get_text_embedding(self, text):
        return self.model.encode(text, convert_to_numpy=True)

    def _get_query_embedding(self, query):
        return self._get_text_embedding(query)

    def _get_text_embeddings(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def _get_query_embeddings(self, queries):
        return self._get_text_embeddings(queries)

    async def _aget_query_embedding(self, query):
        return self._get_query_embedding(query)