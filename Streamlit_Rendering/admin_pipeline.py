# Streamlit_Rendering/admin_pipeline.py

import json
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any # 호환성을 위해 추가

from Streamlit_Rendering.crawl import fetch_article_from_url
from Streamlit_Rendering import repo
# summary.py에서 클래스와 더미 함수 가져오기
from Streamlit_Rendering.summary import FastKoBertSummarizer, summarize_text_dummy
from Streamlit_Rendering.trust import score_trust_dummy

# --------------------------------------------------------------------------
# 1. 모델 캐싱
# --------------------------------------------------------------------------
@st.cache_resource
def load_summarizer_model():
    """
    FastKoBertSummarizer 모델을 메모리에 한 번만 올리고 재사용합니다.
    """
    print("🚀 Loading FastKoBertSummarizer... (First time only)")
    model = FastKoBertSummarizer()
    return model

# --------------------------------------------------------------------------
# 2. 메인 로직
# --------------------------------------------------------------------------

ARTICLE_COLUMNS = [
    "article_id", "title", "source", "url", "published_at", "full_text",
    "summary_text", "keywords", "embed_full", "embed_summary",
    "trust_score", "trust_verdict", "trust_reason", "trust_per_criteria",
    "status",
]

def ingest_one_url(url: str, source: str = "manual", dedup_by_url: bool = True) -> Dict[str, Any]:
    """
    URL 1개 → 크롤링 → (중복 필터링) → 모델 분석 → DB 적재
    """
    try:
        # 1. 중복 체크
        if dedup_by_url and repo.exists_article_url(url):
            return {"status": "skipped", "message": "이미 DB에 존재하는 URL입니다. (중복 스킵)", "url": url}

        # 2. 크롤링
        df_raw = fetch_article_from_url(url=url, source=source)
        
        # 3. 데이터 가공 및 모델 실행
        df_ready = build_ready_rows(df_raw)

        # 4. DB 적재
        repo.upsert_articles(df_ready)
        return {"status": "inserted", "message": "DB에 1건 적재되었습니다.", "url": url}

    except Exception as e:
        return {"status": "error", "message": f"크롤링/적재 실패: {e}", "url": url}


def build_ready_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    크롤링된 데이터를 받아 모델을 돌려 요약, 키워드, 임베딩을 채워 넣습니다.
    """
    # 캐싱된 모델 불러오기
    model = load_summarizer_model()
    
    rows = []
    for _, r in df_raw.iterrows():
        full_text = str(r["full_text"])
        source = str(r["source"])
        
        try:
            # 모델 분석 (요약, 키워드, 임베딩 등 한 번에 추출)
            summary, keywords, content_emb, keyword_emb, summary_emb, trust_score_model = model.analyze_single(full_text)
            
            # Numpy 배열 -> List 변환 (JSON 저장용)
            # hasattr 체크로 안전하게 변환
            embed_full_list = content_emb.tolist() if hasattr(content_emb, 'tolist') else []
            embed_summary_list = summary_emb.tolist() if hasattr(summary_emb, 'tolist') else []
            
        except Exception as e:
            print(f"❌ Model Analysis Error: {e}")
            summary = summarize_text_dummy(full_text)
            keywords = []
            embed_full_list = []
            embed_summary_list = []
            trust_score_model = 50

        # 신뢰도 평가
        trust_detail = score_trust_dummy(full_text, source=source, low=30, high=100)
        final_trust_score = int(trust_score_model)

        rows.append({
            "article_id": str(r["article_id"]),
            "title": str(r["title"]),
            "source": source,
            "url": str(r["url"]),
            "published_at": str(r["published_at"]),
            "full_text": full_text,

            "summary_text": summary,
            "keywords": json.dumps(keywords, ensure_ascii=False),
            "embed_full": json.dumps(embed_full_list),
            "embed_summary": json.dumps(embed_summary_list),

            "trust_score": final_trust_score,
            "trust_verdict": trust_detail.get("verdict", "uncertain"),
            "trust_reason": trust_detail.get("reason", ""),
            "trust_per_criteria": json.dumps(trust_detail.get("per_criteria", {}), ensure_ascii=False),

            "status": "ready",
        })

    df_ready = pd.DataFrame(rows).reindex(columns=ARTICLE_COLUMNS)
    return df_ready

# --------------------------------------------------------------------------
# 개별 테스트용 함수 (타입 힌트 수정됨)
# --------------------------------------------------------------------------

def run_summary(full_text: str) -> str:
    model = load_summarizer_model()
    summary, _, _, _, _, _ = model.analyze_single(full_text)
    return summary

def run_keywords(full_text: str) -> List[str]: # list[str] -> List[str] 로 변경
    model = load_summarizer_model()
    _, keywords, _, _, _, _ = model.analyze_single(full_text)
    return keywords

def run_embedding(text: str) -> List[float]: # list[float] -> List[float] 로 변경
    model = load_summarizer_model()
    emb = model.get_embedding_batch([text])[0]
    return emb.tolist()

def run_trust(full_text: str, source: str) -> dict:
    return score_trust_dummy(full_text, source=source, low=30, high=100)
