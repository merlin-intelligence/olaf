from __future__ import annotations
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .config import FieldMapping

_STATUS_FIELD = "olaf_status"
_PROCESSED_AT_FIELD = "olaf_processed_at"


class ChunkStore:
    def __init__(self, url: str, collection: str, field_mapping: FieldMapping, api_key: str | None = None):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.fm = field_mapping

    def _to_dict(self, point) -> dict:
        p = point.payload or {}
        text = p.get(self.fm.text, "")
        return {
            "id": str(point.id),
            "doc_id": p.get(self.fm.doc_id, ""),
            "chunk_index": p.get(self.fm.chunk_index, 0),
            "text_preview": text[:200] + ("…" if len(text) > 200 else ""),
            "text_length": len(text),
            "status": p.get(_STATUS_FIELD, "pending"),
            "processed_at": p.get(_PROCESSED_AT_FIELD),
        }

    def _parse_id(self, chunk_id: str) -> int | str:
        try:
            return int(chunk_id)
        except ValueError:
            return chunk_id

    def list_chunks(
        self,
        doc_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int | None = None,
    ) -> list[dict]:
        must: list = []
        if doc_id:
            must.append(FieldCondition(key=self.fm.doc_id, match=MatchValue(value=doc_id)))
        if status and status != "all":
            must.append(FieldCondition(key=_STATUS_FIELD, match=MatchValue(value=status)))

        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=must) if must else None,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        return [self._to_dict(p) for p in points]

    def get_chunk(self, chunk_id: str) -> dict | None:
        results = self.client.retrieve(
            collection_name=self.collection,
            ids=[self._parse_id(chunk_id)],
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        p = results[0].payload or {}
        text = p.get(self.fm.text, "")
        return {
            "id": str(results[0].id),
            "doc_id": p.get(self.fm.doc_id, ""),
            "chunk_index": p.get(self.fm.chunk_index, 0),
            "text": text,
            "status": p.get(_STATUS_FIELD, "pending"),
            "processed_at": p.get(_PROCESSED_AT_FIELD),
        }

    def get_chunks_batch(self, chunk_ids: list[str]) -> list[dict]:
        parsed = [self._parse_id(cid) for cid in chunk_ids]
        results = self.client.retrieve(
            collection_name=self.collection,
            ids=parsed,
            with_payload=True,
            with_vectors=False,
        )
        by_str_id = {str(p.id): p for p in results}
        output = []
        for cid in chunk_ids:
            point = by_str_id.get(cid)
            if point is None:
                output.append({"id": cid, "error": "not found"})
            else:
                p = point.payload or {}
                text = p.get(self.fm.text, "")
                output.append({
                    "id": str(point.id),
                    "doc_id": p.get(self.fm.doc_id, ""),
                    "chunk_index": p.get(self.fm.chunk_index, 0),
                    "text": text,
                    "status": p.get(_STATUS_FIELD, "pending"),
                })
        return output

    def mark_processed(self, chunk_id: str) -> bool:
        try:
            self.client.set_payload(
                collection_name=self.collection,
                payload={
                    _STATUS_FIELD: "processed",
                    _PROCESSED_AT_FIELD: datetime.now(timezone.utc).isoformat(),
                },
                points=[self._parse_id(chunk_id)],
            )
            return True
        except Exception:
            return False

    def count_total(self) -> int:
        return self.client.count(collection_name=self.collection).count

    def count_processed(self) -> int:
        return self.client.count(
            collection_name=self.collection,
            count_filter=Filter(must=[
                FieldCondition(key=_STATUS_FIELD, match=MatchValue(value="processed"))
            ]),
        ).count
