# scripts/diag_pgvector.py
import os, json, sys, traceback
import psycopg
from pgvector.psycopg import register_vector, Vector

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv("ddf/.env"))

# --- 1) ENV 로그 (마스킹) ---
def mask(v, n=4): 
    return (v[:n] + "…") if v else None
print("[ENV] PG_HOST:", os.getenv("PG_HOST"))
print("[ENV] PG_DB:",   os.getenv("PG_DB"))
print("[ENV] PG_USER:", os.getenv("PG_USER"))
print("[ENV] OPENAI:",  "set" if os.getenv("OPENAI_API_KEY") else "missing")

# --- 2) DB 연결 + 기본 체크 ---
try:
    with psycopg.connect(
        host=os.getenv("PG_HOST"),
        port=int(os.getenv("PG_PORT", "5432")),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        connect_timeout=5,
    ) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            print("[DB] version:", cur.fetchone()[0])

            # 확장/테이블/차원
            cur.execute("SELECT extname FROM pg_extension WHERE extname='vector'")
            print("[DB] pgvector installed:", bool(cur.fetchone()))

            cur.execute("SELECT to_regclass('public.rag_chunk')")
            print("[DB] rag_chunk exists:", cur.fetchone()[0] is not None)

            cur.execute("SELECT count(*) FROM rag_chunk")
            print("[DB] rag_chunk rows:", cur.fetchone()[0])

            # 임의 한 행의 차원 확인
            cur.execute("SELECT vector_dims(embedding) FROM rag_chunk LIMIT 1")
            row = cur.fetchone()
            print("[DB] embedding dims:", row[0] if row else None)

            # 간단한 kNN 테스트 (임의 쿼리 벡터)
            from langchain_openai import OpenAIEmbeddings
            emb_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
            e = OpenAIEmbeddings(model=emb_model)
            q_vec = Vector(e.embed_query("혈당"))

            sql = """
              SELECT chunk_id, doc_id, left(content,120), (embedding <=> %s) AS dist
                FROM rag_chunk
            ORDER BY embedding <=> %s
               LIMIT 3
            """
            cur.execute(sql, [q_vec, q_vec])
            rows = cur.fetchall()
            print("[DB] top3:")
            for r in rows:
                print("  -", r)
except Exception as e:
    print("[ERROR]", repr(e))
    traceback.print_exc()
    sys.exit(1)

print("[OK] DB & pgvector quick test passed.")


# scripts/diag_retriever.py
import os, traceback
from pg_retriever import PgVectorRetriever

try:
    r = PgVectorRetriever(k=4, min_score=None)  # filter_doc_ids=None
    docs = r.get_relevant_documents("혈당 조절 식단 가이드")
    print("[RET] got", len(docs), "docs")
    for i, d in enumerate(docs, 1):
        print(f"--- Doc {i} ---")
        print("meta:", d.metadata)
        print(d.page_content[:200].replace("\n"," "))
except Exception as e:
    print("[RET][ERROR]", repr(e))
    traceback.print_exc()
