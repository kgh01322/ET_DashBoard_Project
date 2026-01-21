# Streamlit_Rendering/admin_pipeline.py

import json
import pandas as pd
import streamlit as st # Streamlit 기능 사용을 위해 import
from keybert import KeyBERT # 키워드 추출용

from Streamlit_Rendering.crawl import fetch_article_from_url
from Streamlit_Rendering import repo
# summary.py에서 실제 모델 클래스를 가져옵니다.
from Streamlit_Rendering.summary import FastKoBertSummarizer, summarize_text_dummy
from Streamlit_Rendering.trust import score_trust_dummy

# --------------------------------------------------------------------------
# [핵심] 모델 캐싱: Streamlit이 모델을 한 번만 로드하도록 설정
# --------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """
    이 함수는 앱이 실행될 때 딱 한 번만 실행되어 모델을 메모리에 올립니다.
    """
    print("Loading AI Models... (This happens only once)")
    
    # 1. 요약 모델 로드
    summarizer = FastKoBertSummarizer() 
    
    # 2. 키워드 추출 모델 로드 (KeyBERT)
    # 한국어 처리에 좋은 다국어 모델을 사용합니다.
    kw_model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')
    
    return summarizer, kw_model

# --------------------------------------------------------------------------
# 실제 실행 함수들
# --------------------------------------------------------------------------

ARTICLE_COLUMNS = [
    "article_id", "title", "source", "url", "published_at", "full_text",
    "summary_text", "keywords", "embed_full", "embed_summary",
    "trust_score", "trust_verdict", "trust_reason", "trust_per_criteria",
    "status",
]

def ingest_one_url(url: str, source: str = "manual", dedup_by_url: bool = True) -> dict:
    """
    URL 1개 → 크롤링 → (중복 필터링) → DB 적재
    """
    try:
        if dedup_by_url and repo.exists_article_url(url):
            return {"status": "skipped", "message": "이미 DB에 존재하는 URL입니다. (중복 스킵)", "url": url}

        df_raw = fetch_article_from_url(url=url, source=source)
        df_ready = build_ready_rows(df_raw) # 여기서 실제 모델을 돌립니다.

        repo.upsert_articles(df_ready)
        return {"status": "inserted", "message": "DB에 1건 적재되었습니다.", "url": url}

    except Exception as e:
        return {"status": "error", "message": f"크롤링/적재 실패: {e}", "url": url}


def run_summary(full_text: str) -> str:
    """
    캐싱된 모델을 불러와서 실제 요약을 수행합니다.
    """
    if not full_text:
        return ""
    
    try:
        # 1. 모델 가져오기 (캐시된 것 사용)
        summarizer, _ = load_models()
        
        # 2. 요약 수행 (summary.py의 클래스 메서드 이름이 summarize라고 가정)
        # 만약 메서드 이름이 다르면 summarizer.메서드명(full_text)로 바꿔주세요.
        summary = summarizer.summarize(full_text) 
        
        return summary
    except Exception as e:
        print(f"Summary Error: {e}")
        return summarize_text_dummy(full_text, max_chars=100) # 에러 시 더미 반환


def run_keywords(full_text: str) -> list[str]:
    """
    KeyBERT를 이용해 키워드 5개를 추출합니다.
    """
    if not full_text:
        return []
        
    try:
        # 1. 모델 가져오기 (캐시된 것 사용)
        _, kw_model = load_models()
        
        # 2. 키워드 추출
        keywords_tuple = kw_model.extract_keywords(
            full_text, 
            keyphrase_ngram_range=(1, 1), 
            stop_words=None, 
            top_n=5
        )
        # 결과가 [('키워드', 0.5), ...] 형태이므로 단어만 추출
        return [k[0] for k in keywords_tuple]
        
    except Exception as e:
        print(f"Keyword Error: {e}")
        return []


def run_embedding(text: str) -> list[float]:
    # 임베딩은 아직 구현하지 않음 (나중에 필요하면 SentenceTransformer 추가)
    return []


def run_trust(full_text: str, source: str) -> dict:
    return score_trust_dummy(full_text, source=source, low=30, high=100)


def build_ready_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_raw.iterrows():
        full_text = str(r["full_text"])
        source = str(r["source"])

        # [변경] 더미 함수 대신 실제 모델 실행 함수 호출
        summary_text = run_summary(full_text)
        keywords_list = run_keywords(full_text)
        
        trust = run_trust(full_text, source)

        rows.append({
            "article_id": str(r["article_id"]),
            "title": str(r["title"]),
            "source": source,
            "url": str(r["url"]),
            "published_at": str(r["published_at"]),
            "full_text": full_text,

            "summary_text": summary_text,
            "keywords": json.dumps(keywords_list, ensure_ascii=False), # 리스트를 JSON 문자열로 변환
            "embed_full": json.dumps([]),
            "embed_summary": json.dumps([]),

            "trust_score": int(trust.get("score", 50)),
            "trust_verdict": trust.get("verdict", "uncertain"),
            "trust_reason": trust.get("reason", ""),
            "trust_per_criteria": json.dumps(trust.get("per_criteria", {}), ensure_ascii=False),

            "status": "ready",
        })

    df_ready = pd.DataFrame(rows).reindex(columns=ARTICLE_COLUMNS)
    return df_ready
