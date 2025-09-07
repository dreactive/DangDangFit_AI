# app/db.py
import os
from dotenv import load_dotenv, find_dotenv
import psycopg
from pgvector.psycopg import register_vector

load_dotenv(find_dotenv("ddf/.env"))  # 경로는 프로젝트 구조에 맞게

def get_conn():
    host = os.getenv("PG_HOST")
    port = int(os.getenv("PG_PORT"))
    db   = os.getenv("PG_DB")
    user = os.getenv("PG_USER")
    pwd  = os.getenv("PG_PASSWORD")

    if not all([host, db, user, pwd]):
        raise RuntimeError("DB env vars missing: PG_HOST/PG_DB/PG_USER/PG_PASSWORD")

    conn = psycopg.connect(
        host=host, port=port, dbname=db, user=user, password=pwd,
        # sslmode="require",  # 필요하면 사용
    )
    register_vector(conn)
    return conn
