
# 📰 Explainable Fake News Detection & Recommendation Platform (MVP)

본 프로젝트는 **Streamlit + DuckDB 기반의 설명 가능한 뉴스 추천·신뢰도 분석 플랫폼**입니다.
현재 단계에서는 **모델링 포맷 확정 이전**을 전제로,
👉 **크롤링 → DB 적재 → UI 조회 흐름을 안정적으로 구축하는 것**을 목표로 합니다.

---

## 1. 프로젝트 개요

### 🎯 목표

* 관리자 버튼을 통해 **뉴스 크롤링 → DB 적재**
* 사용자 페이지에서는 **DB에 저장된 기사만 조회**
* 요약/신뢰도/추천 모델은 **관리자 파이프라인에서만 계산**
* UI, DB, 크롤링, 모델링을 **파일 단위로 명확히 분리**

### 🧠 현재 단계 (MVP)

* 크롤링: 더미 크롤러 (`crawl.py`)
* 요약/신뢰도: 더미 구현 (`summary.py`, `trust.py`)
* DB: DuckDB (파일 기반)
* UI: Streamlit

---

## 2. 실행 방법

### 2-1. 환경 설정

```bash
pip install streamlit duckdb pandas numpy
```

(모델링 더미 단계이므로 Torch, KoBERT 등은 아직 필수 아님)

---

### 2-2. 실행

```bash
streamlit run app.py
```

브라우저가 열리면:

1. **Manage Mode** 클릭
2. **크롤링 실행 버튼 클릭**
3. 더미 뉴스가 DB에 적재됨
4. 사용자 페이지에서 기사 조회 가능

---

## 3. 프로젝트 디렉토리 구조

```text
.
├─ app.py                     # Streamlit 엔트리 포인트
│
├─ Streamlit_Rendering/
│   ├─ __init__.py
│   │
│   ├─ function.py             # UI 렌더링 (사용자 / 관리자 페이지)
│   ├─ admin_pipeline.py       # (**중요**) 관리자 파이프라인 오케스트레이션
│   ├─ crawl.py                # 뉴스 크롤링 전담 (더미 → 실전 교체 예정)
│   │
│   ├─ repo.py                 # DuckDB 입출력 전담
│   ├─ data.py                 # MOCK 데이터
│   │
│   ├─ summary.py              # 요약 모델 (현재 더미)
│   ├─ trust.py                # 신뢰도 모델 (현재 더미)
│   └─ recommender.py          # 추천 로직 (추후 확장)
│
└─ README.md
```

---

## 4. 파일별 역할 및 주요 함수 설명

---

### 4.1 `app.py`

**역할**

* Streamlit 앱의 진입점
* 세션 상태 기반 페이지 라우팅

**주요 역할**

* DB 초기화 (`repo.init_db()`)
* 관리자 / 사용자 페이지 분기

---

### 4.2 `function.py` (UI 레이어)

**역할**

* Streamlit 화면 렌더링 전담
* DB 조회만 수행 (모델/크롤링 호출 ❌)

#### 주요 함수

```python
render_main_page()
```

* 사용자 메인 페이지
* 검색창, 기사 리스트 표시

```python
render_detail_page(article_id)
```

* 기사 상세 페이지
* **본문만 메인에 표시**
* 요약/신뢰도/피드백은 dialog 팝업으로 제공

```python
render_admin_page()
```

* 관리자 페이지
* 크롤링 버튼 제공
* DB 적재 현황 확인

---

### 4.3 `admin_pipeline.py` (관리자 파이프라인)

**역할**

* 관리자 버튼이 호출하는 **단일 파이프라인**
* 크롤링 → (추후 모델링) → DB 적재

> ⚠️ UI에서는 이 파일만 호출하며,
> 크롤링/모델링 구현은 모두 외부 파일에 위임합니다.

#### 주요 함수

```python
crawl_latest_articles()
```

* `crawl.py`의 크롤러를 호출
* 반환값: raw 기사 DataFrame

```python
build_ready_rows(df_raw)
```

* raw 기사 → DB 스키마 변환
* 현재는 요약/신뢰도 더미값 사용

---

### 4.4 `crawl.py` (크롤링 전담)

**역할**

* 뉴스 수집 로직 전담
* 현재는 더미 구현
* **실 크롤러(RSS / API)로 교체 예정**

#### 주요 함수

```python
crawl_latest_articles_dummy(limit=20)
```

* MOCK 데이터를 기반으로 기사 DataFrame 생성
* 반환 컬럼:

  * `article_id`
  * `title`
  * `source`
  * `url`
  * `published_at`
  * `full_text`

> ⚠️ 이 반환 포맷은 **절대 변경하지 말 것**
> (admin_pipeline과의 계약 인터페이스)

---

### 4.5 `repo.py` (DB 레이어)

**역할**

* DuckDB 입출력 전담
* SQL / 테이블 관리

#### 주요 함수

```python
init_db()
```

* DuckDB 파일 생성
* articles 테이블 초기화

```python
upsert_articles(df)
```

* DataFrame → DuckDB `INSERT OR REPLACE`

```python
load_articles()
```

* DB에 저장된 기사 조회

---

### 4.6 `summary.py` / `trust.py` (모델 레이어)

**역할**

* 요약 / 신뢰도 모델 구현 위치
* **현재는 더미**
* 관리자 파이프라인에서만 사용

#### 예시

```python
summarize_text_dummy(text, max_chars=50)
```

```python
score_trust_dummy(text, source, low=30, high=100)
```

> 모델링 포맷 확정 후 실제 모델로 교체 예정

---

### 4.7 `data.py`

**역할**

* MOCK 뉴스 데이터 보관
* 초기 테스트 및 더미 크롤러용

---

## 5. 설계 철학 (중요)

### ✅ UI / 크롤링 / 모델링 / DB 분리

* UI에서는 **절대 모델이나 크롤러를 직접 호출하지 않음**
* admin_pipeline이 모든 흐름을 조율

### ✅ 서버 없는 구조

* Streamlit 실행 프로세스 자체가 서버 역할
* DuckDB는 파일 기반 DB

### ✅ 확장 가능성

* 추후 FastAPI / 서버 환경으로 옮겨도

  * `crawl.py`
  * `admin_pipeline.py`
  * `repo.py`
    그대로 재사용 가능

---

## 6. 향후 확장 예정

* RSS 기반 실 크롤러
* 요약/신뢰도 모델 실제 연결
* 추천 알고리즘(`recommender.py`) 활성화
* ingest_runs 테이블 추가 (파이프라인 로그)
* PostgreSQL 등 외부 DB 이전

---

## 7. 팀 협업 가이드

| 역할       | 수정 파일                          |
| -------- | ------------------------------ |
| 크롤링      | `crawl.py`                     |
| 모델링      | `summary.py`, `trust.py`       |
| DB/파이프라인 | `repo.py`, `admin_pipeline.py` |
| UI/PM    | `function.py`, `app.py`        |

---

