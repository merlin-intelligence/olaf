from __future__ import annotations
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams
import uuid

# Shared across all EmbeddingService instances in the same process.
# fastembed model loading (~500 MB download + model init) happens once.
_model_cache: dict[str, Any] = {}

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
        self._known_collections: set[str] = set()
        self._ensure_collection()

    def _model_instance(self):
        if self.model_name not in _model_cache:
            try:
                from fastembed import TextEmbedding
                _model_cache[self.model_name] = ("fastembed", TextEmbedding(self.model_name))
            except Exception:
                try:
                    from sentence_transformers import SentenceTransformer
                    _model_cache[self.model_name] = ("sentence_transformers", SentenceTransformer(self.model_name))
                except Exception as e:
                    raise RuntimeError(f"Could not load embedding model {self.model_name}: {e}") from e
        return _model_cache[self.model_name]

    def _vector_size(self) -> int:
        if self.model_name in _VECTOR_SIZE:
            return _VECTOR_SIZE[self.model_name]
        return len(self._embed("probe"))

    def _ensure_collection(self) -> None:
        if self.collection in self._known_collections:
            return
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self._vector_size(), distance=Distance.COSINE),
            )
        self._known_collections.add(self.collection)

    def switch_collection(self, collection_name: str) -> None:
        self.collection = collection_name
        self._ensure_collection()

    def _embed(self, text: str, is_query: bool = False) -> list[float]:
        kind, model = self._model_instance()
        if kind == "fastembed":
            return list(next(iter(model.embed([text]))).tolist())
        prefix = ("query: " if is_query else "passage: ") if "e5" in self.model_name.lower() else ""
        return model.encode(f"{prefix}{text}").tolist()

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
        vector = self._embed(text, is_query=False)
        payload: dict = {"uri": uri, "label": label, "definition": definition}
        if source_chunk_id is not None:
            payload["source_chunk_id"] = source_chunk_id
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=self._concept_point_id(uri), vector=vector, payload=payload)],
        )

    def search_concepts(self, query: str, top_k: int = 10) -> list[dict]:
        vector = self._embed(query, is_query=True)
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        else:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )
        return [
            {
                "uri": h.payload.get("uri", "") if h.payload else "",
                "label": h.payload.get("label", "") if h.payload else "",
                "definition": h.payload.get("definition", "") if h.payload else "",
                "source_chunk_id": h.payload.get("source_chunk_id") if h.payload else None,
                "score": round(h.score, 4) if hasattr(h, "score") and h.score is not None else 0.0,
                "match_type": "semantic",
            }
            for h in hits
        ]

    def delete_concept(self, uri: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[self._concept_point_id(uri)]),
        )
