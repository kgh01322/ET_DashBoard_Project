import json
import pandas as pd
import numpy as np

# 1. 모델 파일 가져오기
from Streamlit_Rendering.nlp_engine import FastKoBertSummarizer
from Streamlit_Rendering.crawl import fetch_article_from_url
from Streamlit_Rendering import repo
from Streamlit_Rendering.trust import score_trust_dummy  # 기존 더미 유지하거나 모델로 대체 가능

# --- 전역 모델 인스턴스 (최초 1회 로딩) ---
ANALYZER = None

def get_analyzer():
    global ANALYZER
    if ANALYZER is None:
        ANALYZER = FastKoBertSummarizer()
    return ANALYZER

ARTICLE_COLUMNS = [
    "article_id", "title", "source", "url", "published_at", "full_text",
    "summary_text", "keywords", "embed_full", "embed_summary",
    "trust_score", "trust_verdict", "trust_reason", "trust_per_criteria",
    "status",
]

def ingest_one_url(url: str, source: str = "manual", dedup_by_url: bool = True) -> dict:
    try:
        if dedup_by_url and repo.exists_article_url(url):
            return {"status": "skipped", "message": "이미 DB에 존재하는 URL입니다.", "url": url}

        df_raw = fetch_article_from_url(url=url, source=source)
        df_ready = build_ready_rows(df_raw) # 여기서 아래 함수들이 실행됨

        repo.upsert_articles(df_ready)
        return {"status": "inserted", "message": "DB에 1건 적재되었습니다.", "url": url}

    except Exception as e:
        return {"status": "error", "message": f"크롤링/적재 실패: {e}", "url": url}


## ================================================================================
def run_summary(full_text: str):
    """
    반환값: (summary_text, content_emb_json, summary_emb_json)
    """
    analyzer = get_analyzer()
    summary, _, c_emb, _, s_emb, _ = analyzer.analyze_single(full_text)
    
    # DB 저장을 위해 Numpy 배열을 JSON 문자열로 변환
    def to_json(emb): 
        return json.dumps(emb.tolist()) if hasattr(emb, 'tolist') else "[]"

    return summary, to_json(c_emb), to_json(s_emb)

def run_keywords(full_text: str):
    """
    반환값: (keywords_json, keyword_emb_json)
    """
    analyzer = get_analyzer()
    # 이미 run_summary에서 모델이 돌았겠지만, 구조상 여기서 다시 돌립니다.
    _, keywords, _, k_emb, _, _ = analyzer.analyze_single(full_text)
    
    def to_json(emb): 
        return json.dumps(emb.tolist()) if hasattr(emb, 'tolist') else "[]"
    
    # 키워드 리스트도 JSON 문자열로 변환
    return json.dumps(keywords, ensure_ascii=False), to_json(k_emb)
## ================================================================================

def run_trust(full_text: str, source: str) -> dict:

    return score_trust_dummy(full_text, source=source, low=30, high=100)

# =========================================================
# 데이터 조립 함수 (리턴값을 받아서 풀도록 수정)
# =========================================================

def build_ready_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_raw.iterrows():
        full_text = str(r["full_text"])
        source = str(r["source"])

        # 1. run_summary가 (요약문, 본문임베딩, 요약임베딩) 3개를 반환하므로 언패킹
        summary_text, embed_full, embed_summary = run_summary(full_text)
        
        # 2. run_keywords가 (키워드JSON, 키워드임베딩) 2개를 반환하므로 언패킹
        keywords_json, _ = run_keywords(full_text) 
        # (참고: DB 스키마에 'keyword_embedding' 컬럼이 없다면 여기서 버리거나 스키마 추가 필요)

        # 3. 신뢰도
        trust = run_trust(full_text)

        rows.append({
            "article_id": str(r["article_id"]),
            "title": str(r["title"]),
            "source": source,
            "url": str(r["url"]),
            "published_at": str(r["published_at"]),
            "full_text": full_text,

            # 위에서 받은 값들 매핑
            "summary_text": summary_text,
            "keywords": keywords_json,
            "embed_full": embed_full,       # JSON 문자열
            "embed_summary": embed_summary, # JSON 문자열

            "trust_score": trust.get("score", 50),
            "trust_verdict": trust.get("verdict", "uncertain"),
            "trust_reason": trust.get("reason", ""),
            "trust_per_criteria": json.dumps(trust.get("per_criteria", {}), ensure_ascii=False),

            "status": "ready",
        })

    df_ready = pd.DataFrame(rows).reindex(columns=ARTICLE_COLUMNS)
    return df_ready
