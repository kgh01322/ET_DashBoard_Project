# Streamlit_Rendering/trust.py
import numpy as np

def score_trust_dummy(text: str, source: str | None = None, low: int = 30, high: int = 100) -> dict:
    """
    더미 신뢰도 함수:
    - 30~100 사이의 점수를 무작위로 반환합니다.
    - 반환 포맷은 추후 TELLER/LLM 구현 시 그대로 확장 가능합니다.
    """
    score = int(np.random.randint(low, high + 1))

    # 매우 단순한 verdict 규칙(더미)
    if score >= 70:
        verdict = "likely_true"
    elif score >= 40:
        verdict = "uncertain"
    else:
        verdict = "likely_false"

    return {
        "score": score,
        "verdict": verdict,
        "reason": "더미 신뢰도 점수입니다.",
        "per_criteria": {
            "source_credibility": {"score": 1, "reason": "더미"},
            "evidence_support": {"score": 1, "reason": "더미"},
            "style_neutrality": {"score": 1, "reason": "더미"},
            "logical_consistency": {"score": 1, "reason": "더미"},
            "clickbait_risk": {"score": 1, "reason": "더미"},
        }
    }
