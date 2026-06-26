from __future__ import annotations
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams
import uuid

_VECTOR_SIZE: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
}


class EmbeddingService:
    def __init__(self, model_name: str, client: QdrantClient, collection: str):
        self.model_name = model_name
        self.client = client
        self.collection = collection
        self._model = None
        self._ensure_collection()

    def _model_instance(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self.model_name)
        return self._model

    def _vector_size(self) -> int:
        if self.model_name in _VECTOR_SIZE:
            return _VECTOR_SIZE[self.model_name]
        # Unknown model: embed a probe string to determine size
        return len(self._embed("probe"))

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self._vector_size(), distance=Distance.COSINE),
            )

    def switch_collection(self, collection_name: str) -> None:
        self.collection = collection_name
        self._ensure_collection()

    def _embed(self, text: str) -> list[float]:
        return list(next(iter(self._model_instance().embed([text]))).tolist())

    def _concept_point_id(self, uri: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, uri))

    def upsert_concept(
        self,
        uri: str,
        label: str,
        definition: str,
        source_chunk_id: str | None = None,
    ) -> None:
        text = f"{label}. {definition}" if definition else label
        vector = self._embed(text)
        payload: dict = {"uri": uri, "label": label, "definition": definition}
        if source_chunk_id is not None:
            payload["source_chunk_id"] = source_chunk_id
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=self._concept_point_id(uri), vector=vector, payload=payload)],
        )

    def search_concepts(self, query: str, top_k: int = 10) -> list[dict]:
        vector = self._embed(query)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "uri": h.payload.get("uri", ""),
                "label": h.payload.get("label", ""),
                "definition": h.payload.get("definition", ""),
                "source_chunk_id": h.payload.get("source_chunk_id"),
                "score": round(h.score, 4),
                "match_type": "semantic",
            }
            for h in hits
        ]

    def delete_concept(self, uri: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[self._concept_point_id(uri)]),
        )
