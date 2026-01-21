# Streamlit_Rendering/admin_pipeline.py

import json
import pandas as pd
import numpy as np
import streamlit as st

from Streamlit_Rendering.crawl import fetch_article_from_url
from Streamlit_Rendering import repo
# summary.py에서 클래스와 더미 함수 가져오기
from Streamlit_Rendering.summary import FastKoBertSummarizer, summarize_text_dummy
from Streamlit_Rendering.trust import score_trust_dummy

# --------------------------------------------------------------------------
# 1. 모델 캐싱 (가장 중요!)
# Streamlit은 새로고침할 때마다 코드를 다시 실행하는데, 
# 모델 로딩을 매번 하면 서버가 터집니다. 이를 방지하는 코드입니다.
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

def ingest_one_url(url: str, source: str = "manual", dedup_by_url: bool = True) -> dict:
    """
    URL 1개 → 크롤링 → (중복 필터링) → 모델 분석 → DB 적재
    """
    try:
        # 1. 중복 체크
        if dedup_by_url and repo.exists_article_url(url):
            return {"status": "skipped", "message": "이미 DB에 존재하는 URL입니다. (중복 스킵)", "url": url}

        # 2. 크롤링
        df_raw = fetch_article_from_url(url=url, source=source)
        
        # 3. 데이터 가공 및 모델 실행 (여기가 핵심)
        df_ready = build_ready_rows(df_raw)

        # 4. DB 적재
        repo.upsert_articles(df_ready)
        return {"status": "inserted", "message": "DB에 1건 적재되었습니다.", "url": url}

    except Exception as e:
        return {"status": "error", "message": f"크롤링/적재 실패: {e}", "url": url}


def build_ready_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    크롤링된 데이터를 받아 모델(FastKoBertSummarizer)을 돌려 
    요약, 키워드, 임베딩을 채워 넣습니다.
    """
    # 캐싱된 모델 불러오기
    model = load_summarizer_model()
    
    rows = []
    for _, r in df_raw.iterrows():
        full_text = str(r["full_text"])
        source = str(r["source"])
        
        # -------------------------------------------------------
        # [핵심] 모델 analyze_single 메서드 한 번 호출로 모든 값 획득
        # 반환값 순서: summary, keywords, content_emb, keyword_emb, summary_emb, trust_score
        # -------------------------------------------------------
        try:
            summary, keywords, content_emb, keyword_emb, summary_emb, trust_score_model = model.analyze_single(full_text)
            
            # Numpy 배열을 JSON 저장을 위해 리스트로 변환
            embed_full_list = content_emb.tolist() if hasattr(content_emb, 'tolist') else []
            embed_summary_list = summary_emb.tolist() if hasattr(summary_emb, 'tolist') else []
            
        except Exception as e:
            print(f"❌ Model Analysis Error: {e}")
            # 에러 발생 시 더미 값으로 대체
            summary = summarize_text_dummy(full_text)
            keywords = []
            embed_full_list = []
            embed_summary_list = []
            trust_score_model = 50

        # 신뢰도 상세 평가 (Trust 로직은 별도 함수와 병행 사용)
        trust_detail = score_trust_dummy(full_text, source=source, low=30, high=100)
        
        # 모델 점수와 룰베이스 점수 중 모델 점수를 우선하거나 평균을 낼 수 있음
        # 여기서는 모델 점수를 우선으로 넣음
        final_trust_score = int(trust_score_model)

        rows.append({
            "article_id": str(r["article_id"]),
            "title": str(r["title"]),
            "source": source,
            "url": str(r["url"]),
            "published_at": str(r["published_at"]),
            "full_text": full_text,

            # 모델 분석 결과 매핑
            "summary_text": summary,
            "keywords": json.dumps(keywords, ensure_ascii=False), # 리스트 -> JSON 문자열
            "embed_full": json.dumps(embed_full_list),            # 리스트 -> JSON 문자열
            "embed_summary": json.dumps(embed_summary_list),      # 리스트 -> JSON 문자열

            # 신뢰도 정보
            "trust_score": final_trust_score,
            "trust_verdict": trust_detail.get("verdict", "uncertain"),
            "trust_reason": trust_detail.get("reason", ""),
            "trust_per_criteria": json.dumps(trust_detail.get("per_criteria", {}), ensure_ascii=False),

            "status": "ready",
        })

    df_ready = pd.DataFrame(rows).reindex(columns=ARTICLE_COLUMNS)
    return df_ready

# --------------------------------------------------------------------------
# 개별 테스트용 함수 (필요한 경우에만 사용)
# build_ready_rows에서 이미 다 처리하므로 실제 파이프라인에서는 잘 안 쓰임
# --------------------------------------------------------------------------

def run_summary(full_text: str) -> str:
    model = load_summarizer_model()
    summary, _, _, _, _, _ = model.analyze_single(full_text)
    return summary

def run_keywords(full_text: str) -> list[str]:
    model = load_summarizer_model()
    _, keywords, _, _, _, _ = model.analyze_single(full_text)
    return keywords

def run_embedding(text: str) -> list[float]:
    model = load_summarizer_model()
    # 임베딩만 필요할 때 (Batch 처리 활용)
    emb = model.get_embedding_batch([text])[0]
    return emb.tolist()

def run_trust(full_text: str, source: str
