# Streamlit_Rendering/crawl.py
import pandas as pd
from datetime import datetime, timedelta
from Streamlit_Rendering.data import MOCK_DB


def crawl_latest_articles_dummy(limit: int = 10) -> pd.DataFrame:
    """
    더미 크롤러
    - 실제 크롤러로 교체해도 인터페이스 유지
    - 반환 DataFrame 컬럼은 'raw 기사 스키마'
    """

    rows = []
    base_time = datetime.now()

    for i, (_, a) in enumerate(MOCK_DB.items()):
        if i >= limit:
            break

        rows.append({
            "article_id": str(a.get("id", i)),
            "title": a.get("title", ""),
            "source": a.get("source", "dummy_source"),
            "url": a.get("url", ""),
            "published_at": (base_time - timedelta(minutes=i)).isoformat(),
            "full_text": a.get("full_text", ""),
        })

    return pd.DataFrame(rows)
