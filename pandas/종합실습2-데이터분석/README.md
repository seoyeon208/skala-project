# Stack Overflow 2024 — AI 태도와 개발자 연봉 분석

Stack Overflow Developer Survey 2024 데이터로 **"AI 도구에 대한 태도(`AISent`)가 개발자 연봉과 어떤 관계를 갖는가"** 를 분석하는 파이프라인입니다.

로딩 → 전처리 → EDA → 시각화 → 통계 검정 → ML → 심층분석 → 보고서 생성까지 한 번에 실행되며, 결과는 `report.md`로 자동 정리됩니다.

## 요구 사항

- Python 3.9+
- 원본 데이터 `results.csv` (Stack Overflow Developer Survey 2024, 프로젝트 루트에 위치)
  - 출처: https://github.com/StackExchange/Survey/tree/main/packages/archive/2024

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

```bash
python3 -m src.main                       # 기본 (results.csv 사용)
python3 -m src.main --data-path results.csv
```

실행이 끝나면:
- `report.md` — 분석 결과 종합 보고서
- `outputs/charts/` — Seaborn 정적 차트, Plotly 인터랙티브 차트
- `outputs/models/` — 학습된 모델(`joblib`)

## 파이프라인 단계

각 단계는 `src/` 아래 동명 모듈이며, `main.py`가 순서대로 호출합니다.

| 단계 | 모듈 | 내용 |
| --- | --- | --- |
| 0. 벤치마크 | `benchmark.py` | Pandas vs Polars 로딩·필터·집계 속도 비교 (부록) |
| 1. 로딩 | `loading.py` | 두 라이브러리로 읽어 결과 정합성(shape·결측) 대조 |
| 2. 전처리 | `preprocess.py` | 결측치·중복·이상치 처리 |
| 3. EDA | `eda.py` | 기술통계, 분포 확인 |
| 4. 시각화 | `viz.py` | Seaborn 정적 + Plotly 인터랙티브 차트 |
| 5. 통계 | `stats.py` | 그룹 간 평균 연봉 차이 가설검정 |
| 6. ML | `model.py` | 연봉 예측 모델 학습·평가 |
| 7. 심층분석 | `deep_dive.py` | 세부 그룹별 추가 분석 |
| 8. 보고서 | `report.py` / `reporting.py` | 위 결과를 모아 `report.md` 조립 |

보조 모듈: `config.py`(경로·상수), `results.py`(결과 dataclass), `formatting.py`(출력 포맷).

## 구조 메모

각 단계는 진행 상황을 콘솔에 출력하는 동시에 결과를 구조화된 dataclass로 반환합니다. 마지막 `render_report()`가 그 결과들을 받아 `report.md`를 조립하므로, **보고서 문구를 바꿔도 분석 로직은 건드릴 필요가 없습니다.**
