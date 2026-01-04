# Streamlit_Rendering/admin_pipeline.py
import json
import pandas as pd

from Streamlit_Rendering.crawl import crawl_latest_articles_dummy
from Streamlit_Rendering.summary import summarize_text_dummy
from Streamlit_Rendering.trust import score_trust_dummy


def crawl_latest_articles() -> pd.DataFrame:
    """
    크롤링 단계
    - 지금은 crawl.py의 더미 크롤러 사용
    - 나중에 실 크롤러로 교체해도 이 함수 시그니처는 유지
    """
    return crawl_latest_articles_dummy(limit=20)


def run_summary(full_text: str) -> str:
    return summarize_text_dummy(full_text, max_chars=50)


def run_keywords(full_text: str) -> list[str]:
    return []


def run_embedding(text: str) -> list[float]:
    return []


def run_trust(full_text: str, source: str) -> dict:
    return score_trust_dummy(full_text, source=source, low=30, high=100)


def build_ready_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []

    required_cols = ["article_id", "title", "source", "url", "published_at", "full_text"]
    for c in required_cols:
        if c not in df_raw.columns:
            raise ValueError(f"df_raw에 필수 컬럼이 없습니다: {c}")

    for _, r in df_raw.iterrows():
        article_id = str(r["article_id"])
        title = str(r["title"])
        source = str(r["source"])
        url = str(r["url"])
        published_at = str(r["published_at"])
        full_text = str(r["full_text"])

        summary_text = run_summary(full_text)
        keywords = run_keywords(full_text)
        embed_full = run_embedding(full_text)
        embed_summary = run_embedding(summary_text)
        trust = run_trust(full_text, source)

        rows.append({
            "article_id": article_id,
            "title": title,
            "source": source,
            "url": url,
            "published_at": published_at,
            "full_text": full_text,

            "summary_text": summary_text,
            "keywords": json.dumps(keywords, ensure_ascii=False),
            "embed_full": json.dumps(embed_full),
            "embed_summary": json.dumps(embed_summary),

            "trust_score": int(trust.get("score", 50)),
            "trust_verdict": trust.get("verdict", "uncertain"),
            "trust_reason": trust.get("reason", ""),
            "trust_per_criteria": json.dumps(
                trust.get("per_criteria", {}), ensure_ascii=False
            ),

            # ⚠️ 모델링 확정 전이므로 ready 대신 pending 권장
            "status": "pending",
        })

    return pd.DataFrame(rows)
