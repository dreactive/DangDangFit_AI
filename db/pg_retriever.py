# app/pg_retriever.py
from __future__ import annotations
from typing import List, Optional, Any, Dict
import os
import psycopg
from pgvector.psycopg import register_vector, Vector

# LangChain 버전 호환 import
try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import BaseRetriever, Document  # old versions

# Pydantic v1/v2 호환
try:
    from pydantic import PrivateAttr, Field
    PydanticV2 = True
except Exception:  # v1
    from pydantic import BaseModel  # dummy import for typing only
    def PrivateAttr(default=None):  # type: ignore
        return default
    class Field:  # type: ignore
        def __init__(self, default=None, **kwargs): pass
    PydanticV2 = False

from langchain_openai import OpenAIEmbeddings

class PgVectorRetriever(BaseRetriever):
    # ====== Pydantic 필드: 클래스 레벨에 선언해야 함 ======
    k: int = Field(4, description="top-k 결과 개수")
    min_score: Optional[float] = Field(default=None, description="유사도 컷오프(0~1)")
    filter_doc_ids: Optional[List[str]] = Field(default=None, description="문서 제한: doc_id 문자열 목록")
    embed_model: str = Field(default="text-embedding-3-small", description="임베딩 모델명")

    pg_host: Optional[str] = Field(default=None)
    pg_port: int = Field(default=int(os.getenv("PG_PORT", "5432")))
    pg_db: Optional[str] = Field(default=None)
    pg_user: Optional[str] = Field(default=None)
    pg_password: Optional[str] = Field(default=None)

    # ====== Pydantic 비직렬 내부 상태: PrivateAttr 사용 ======
    _emb: Optional[OpenAIEmbeddings] = PrivateAttr(default=None)

    # Pydantic v1/v2 모두에서 임의 타입 허용
    if PydanticV2:
        model_config = {"arbitrary_types_allowed": True}
    else:
        class Config:
            arbitrary_types_allowed = True

    # __init__ 오버라이드 금지(가능하면). 필드 값은 생성자 인자로 들어옵니다.
    # 필요 시 __init__을 써도 super().__init__(**data) 먼저 호출하고, 이후엔 PrivateAttr만 설정하세요.

    def __init__(self, **data: Any):
        # env 기본값 주입 (사용자가 명시하지 않으면 .env 값으로 채움)
        data.setdefault("pg_host", os.getenv("PG_HOST"))
        data.setdefault("pg_db",   os.getenv("PG_DB"))
        data.setdefault("pg_user", os.getenv("PG_USER"))
        data.setdefault("pg_password", os.getenv("PG_PASSWORD"))
        super().__init__(**data)
        # PrivateAttr은 자유롭게 세팅 가능
        self._emb = None

    # 내부 준비(필요 시점에만)
    def _ensure_ready(self):
        if not all([self.pg_host, self.pg_db, self.pg_user, self.pg_password]):
            raise RuntimeError("DB env vars missing: PG_HOST/PG_DB/PG_USER/PG_PASSWORD")
        if self._emb is None:
            self._emb = OpenAIEmbeddings(model=self.embed_model)

    def _get_conn(self) -> psycopg.Connection:
        conn = psycopg.connect(
            host=self.pg_host,
            port=self.pg_port,
            dbname=self.pg_db,
            user=self.pg_user,
            password=self.pg_password,
        )
        register_vector(conn)
        return conn

    # 최신 시그니처(run_manager) 호환
    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        self._ensure_ready()

        q_vec = Vector(self._emb.embed_query(query))  # type: ignore

        where, params = [], [q_vec]
        if self.filter_doc_ids:
            # 안전하게 text 비교 (UUID 포맷 아니어도 에러 없이 0매칭)
            where.append("doc_id::text = ANY(%s)")
            params.append(self.filter_doc_ids)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        sql = f"""
          SELECT chunk_id, doc_id, content, meta, (embedding <=> %s) AS distance
            FROM rag_chunk
            {where_sql}
        ORDER BY embedding <=> %s
           LIMIT %s
        """
        params.extend([q_vec, self.k])

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        docs: List[Document] = []
        for chunk_id, doc_id, content, meta, dist in rows:
            score = 1.0 - float(dist)
            if self.min_score is not None and score < self.min_score:
                continue
            md: Dict[str, Any] = {"doc_id": str(doc_id), "chunk_id": str(chunk_id), "score": score}
            if isinstance(meta, dict):
                md.update(meta)
            docs.append(Document(page_content=content, metadata=md))
        return docs
