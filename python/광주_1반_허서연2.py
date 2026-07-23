"""
=======================================================================
 프로그램명  : Python Practice 2
 작  성  자  : 광주_1반_허서연
 작  성  일  : 2026-07-20
 Python 버전 : 3.8+
-----------------------------------------------------------------------
 [프로그램 개요]
   Python_Practice1_Data.json 활용 실습 코드.
   예외처리·파일 읽기, Pydantic v2 검증, CSV/JSON 저장·재로딩 구현

   1) safe_load_csv()   : 예외 처리 + 파일 읽기
   2) SalesRecord       : Pydantic v2 스키마 정의
   3) validate_records(): 검증 파이프라인 (valid / errors 분리)
   4) 결과 파일 저장 + 재로딩 확인
=======================================================================
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import BaseModel, ValidationError, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =======================================================================
# 1) 예외 처리 + 파일 읽기
# =======================================================================
def safe_load_csv(filepath: str) -> Optional[List[dict]]:
    """파일을 읽어 dict 리스트로 반환합니다. 파일 없으면 None."""
    try:
        namespace: dict = {}
        with open(filepath, encoding="utf-8") as f:
            exec(f.read(), namespace)
        data = namespace.get("sales")
        if not isinstance(data, list):
            raise ValueError("'sales' 리스트를 찾을 수 없습니다.")
        logger.info(f"로딩 성공: {len(data)}건")
        return data
    except FileNotFoundError:
        logger.error(f"파일 없음: {filepath}")
        return None
    except Exception as e:
        logger.error(f"로딩 실패: {e}")
        return None
    finally:
        print("로딩 종료")


# =======================================================================
# 2) Pydantic v2 스키마 정의
# =======================================================================
class SalesRecord(BaseModel):
    """판매 레코드 검증 모델 (month·region 필수, amount > 0, category 선택)."""
    month: str = Field(min_length=1)
    region: str = Field(min_length=1)
    amount: float = Field(gt=0, description="양수")
    category: Optional[str] = None


# =======================================================================
# 3) 검증 파이프라인 (valid / errors 분리)
# =======================================================================
def validate_records(raw: List[dict]) -> Tuple[List[SalesRecord], List[dict]]:
    """raw_data를 SalesRecord로 변환해 (valid, errors) 튜플을 반환합니다."""
    valid, errors = [], []
    for row in raw:
        try:
            valid.append(SalesRecord(**row))
        except ValidationError as e:
            msg = e.errors()[0]['msg'].replace("Value error, ", "")
            print(f"  ValidationError — {msg} | {row}")
            errors.append({"row": row, "error": str(e)})
    return valid, errors


# =======================================================================
# 실행
# =======================================================================

# -- 1) safe_load_csv: None 반환 검증 --
result = safe_load_csv("empty_file.csv")
assert result is None, "None 반환 실패"

# -- 1) safe_load_csv: 정상 로딩 --
safe_load_csv("Python_Practice1_Data.json")

# -- 검증용 테스트 데이터 (4건 valid / 3건 errors) --
test_data = [
    {"month": "2024-01", "region": "서울", "amount": 1500, "category": "전자"},   # valid
    {"month": "2024-01", "region": "부산", "amount": 800,  "category": "의류"},   # valid
    {"month": "2024-02", "region": "대구", "amount": 2000},                        # valid
    {"month": "2024-02", "region": "인천", "amount": 500,  "category": "식품"},   # valid
    {"month": "",        "region": "광주", "amount": 700,  "category": "전자"},   # error: month 비어있음
    {"month": "2024-03", "region": "",     "amount": 300},                         # error: region 비어있음
    {"month": "2024-03", "region": "울산", "amount": -100, "category": "의류"},   # error: amount ≤ 0
]

# -- 3) 검증 파이프라인 실행 --
print("\n[검증 파이프라인]")
valid, errors = validate_records(test_data)
assert len(valid) == 4,  f"valid 건수 오류: {len(valid)}"
assert len(errors) == 3, f"errors 건수 오류: {len(errors)}"
print(f"\nvalid: {len(valid)} / errors: {len(errors)}")

# -- 4) 결과 파일 저장 --
VALID_CSV   = "valid_records.csv"
ERRORS_JSON = "errors.json"

with open(VALID_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["month", "region", "amount", "category"])
    writer.writeheader()
    writer.writerows(r.model_dump() for r in valid)

# 오류 리포트 저장
Path(ERRORS_JSON).write_text(json.dumps(errors, ensure_ascii=False))

# -- 4) 재로딩 확인 --
with open(VALID_CSV, encoding="utf-8") as f:
    reloaded = list(csv.DictReader(f))

assert len(reloaded) == 4, f"재로딩 건수 오류: {len(reloaded)}"
print(f"len(reloaded) == {len(reloaded)}")
