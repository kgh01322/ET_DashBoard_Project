# Streamlit_Rendering/repo.py
import duckdb
import pandas as pd

DB_PATH = "app_db.duckdb"

def init_db():
    con = duckdb.connect(DB_PATH)
    con.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        article_id VARCHAR PRIMARY KEY,
        title VARCHAR,
        source VARCHAR,
        url VARCHAR,
        published_at VARCHAR,
        full_text VARCHAR,
        summary_text VARCHAR,
        keywords VARCHAR,
        embed_full VARCHAR,
        embed_summary VARCHAR,
        trust_score INTEGER,
        trust_verdict VARCHAR,
        trust_reason VARCHAR,
        trust_per_criteria VARCHAR,
        status VARCHAR
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS user_events (
        user_id VARCHAR,
        ts VARCHAR,
        event VARCHAR,
        article_id VARCHAR
    );
    """)
    con.close()

def upsert_articles(df: pd.DataFrame):
    con = duckdb.connect(DB_PATH)
    con.register("df", df)
    con.execute("INSERT OR REPLACE INTO articles SELECT * FROM df")
    con.close()

def load_articles(where_sql: str = "") -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    q = "SELECT * FROM articles"
    if where_sql.strip():
        q += f" WHERE {where_sql}"
    df = con.execute(q).fetchdf()
    con.close()
    return df

def append_event(user_id: str, ts: str, event: str, article_id: str):
    con = duckdb.connect(DB_PATH)
    con.execute(
        "INSERT INTO user_events VALUES (?, ?, ?, ?)",
        [user_id, ts, event, article_id]
    )
    con.close()

def load_events(user_id: str) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    df = con.execute(
        "SELECT * FROM user_events WHERE user_id = ? ORDER BY ts DESC",
        [user_id]
    ).fetchdf()
    con.close()
    return df
