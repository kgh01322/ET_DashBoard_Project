# Streamlit_Rendering/recommender.py
import json
import numpy as np
import pandas as pd

def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def base_similarity(df_articles: pd.DataFrame, query_vec: list[float]) -> np.ndarray:
    scores = []
    for _, r in df_articles.iterrows():
        vec = json.loads(r["embed_summary"])
        scores.append(_cosine(query_vec, vec))
    return np.asarray(scores, dtype=float)

def session_profile_vectors(events_df: pd.DataFrame, articles_df: pd.DataFrame, n_recent: int = 5):
    # TODO: 세션 벡터(최근 본 기사 embed_summary 평균), 프로필(키워드 카운트 기반) 등을 구성
    raise NotImplementedError
