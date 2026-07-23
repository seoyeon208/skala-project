"""
=======================================================================
 프로그램명  : Python Practice 1
 작  성  자  : 광주_1반_허서연
 작  성  일  : 2026-07-20
 Python 버전 : 3.8+
-----------------------------------------------------------------------
 [프로그램 개요]
   Python_Practice1_Data.json 의 Sales 데이터를 활용한 실습 코드.

   1) 리스트/딕셔너리 컴프리헨션
      amount ≥ 1000 거래 필터링 · 지역별 총매출 집계

   2) Counter + defaultdict
      지역별 거래 건수(Counter) · 카테고리별 amount 리스트(defaultdict)

   3) 제너레이터 — 메모리 비교
      amount > 1000 제너레이터와 리스트 버전의 sys.getsizeof 비교

   4) 종합 — 월별 카테고리 매출 집계
      (month, category) 복합 키 · defaultdict + 컴프리헨션 조합

=======================================================================
"""

import sys
from collections import Counter, defaultdict
from typing import Iterator, Dict, List


def load_sales(filepath: str) -> List[Dict]:
    """파일에서 sales 리스트를 읽어 반환합니다. 파일 미존재·파싱 오류 시 예외 발생."""
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        namespace: Dict = {}
        exec(source, namespace)          # 파일 안의 'sales = [...]' 실행
        sales_data = namespace.get("sales")
        if not isinstance(sales_data, list):
            raise ValueError("파일에 'sales' 리스트가 없습니다.")
        return sales_data
    except FileNotFoundError:
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {filepath}")
    except SyntaxError as exc:
        raise ValueError(f"파일 파싱 오류 (구문 오류): {exc}") from exc


# 데이터 로딩 — 이후 모든 섹션에서 공통으로 사용
try:
    sales: List[Dict] = load_sales("Python_Practice1_Data.json")
except (FileNotFoundError, ValueError) as e:
    print(f"[오류] {e}")
    sys.exit(1)


# =======================================================================
# 실습 1) 리스트 / 딕셔너리 컴프리헨션
# =======================================================================

# ① amount ≥ 1000 거래 필터링 (리스트 컴프리헨션)
high_value: List[Dict] = [row for row in sales if row["amount"] >= 1000]

# 상위 3건 — sorted(): 원본 유지, 새 리스트 반환 (내림차순)
top3 = sorted(high_value, key=lambda r: r["amount"], reverse=True)[:3]

print("1) amount ≥ 1000 Top 3:")
for i, row in enumerate(top3, 1):
    print(f"  {i}. {row['region']} / {row['category']} / {row['amount']:,}원")

# ② 지역별 총매출 (딕셔너리 컴프리헨션)
#    set 컴프리헨션으로 중복 없는 지역 추출 → 딕셔너리 컴프리헨션으로 집계
regions = {row["region"] for row in sales}   # set 컴프리헨션 (중복 제거)
region_total: Dict[str, int] = {
    region: sum(row["amount"] for row in sales if row["region"] == region)
    for region in regions
}

print("\n   지역별 총매출:")
for region, total in sorted(region_total.items(), key=lambda x: -x[1]):
    print(f"  {region}: {total:,}원")

# assert: region_total 값 정확성 검증
assert region_total["서울"] == sum(r["amount"] for r in sales if r["region"] == "서울"), \
    "서울 총매출 불일치"


# =======================================================================
# 실습 2) Counter + defaultdict
# =======================================================================

# Counter: 지역별 거래 건수
#   Counter(이터러블) → {값: 빈도} 딕셔너리 서브클래스
region_counter: Counter = Counter(row["region"] for row in sales)

print("\n2) 지역별 거래 건수")
for region, count in region_counter.most_common():
    print(f"  {region}: {count}건")

# assert: most_common() 내림차순 정렬 순서 검증
assert all(
    region_counter.most_common()[i][1] >= region_counter.most_common()[i + 1][1]
    for i in range(len(region_counter) - 1)
), "most_common 정렬 순서 오류"

# defaultdict: 카테고리별 amount 리스트 그룹핑
#   defaultdict(list) → 새 키 접근 시 자동으로 빈 리스트 생성 (키 체크 불필요)
category_amounts: defaultdict = defaultdict(list)
for row in sales:
    category_amounts[row["category"]].append(row["amount"])

print("\n카테고리별 amount")
for category, amounts in sorted(category_amounts.items()):
    print(f"  {category}: 합계:{sum(amounts):,}원, 평균:{sum(amounts)/len(amounts):,.1f}원, {len(amounts)}건")


# =======================================================================
# 실습 3 — 제너레이터 - 메모리 비교
# =======================================================================

def high_amount_generator(data: List[Dict], threshold: int = 1000) -> Iterator[Dict]:
    """amount > threshold 인 행만 하나씩 yield 하는 제너레이터 함수."""
    for row in data:
        if row["amount"] > threshold:
            yield row   # 메모리에 전체를 올리지 않고 하나씩 반환


# 제너레이터
gen_obj = high_amount_generator(sales, threshold=1000)
# 리스트 버전
list_ver: List[Dict] = [row for row in sales if row["amount"] > 1000]

# sys.getsizeof: 객체 자체의 얕은(shallow) 크기 반환
gen_size  = sys.getsizeof(gen_obj)
list_size = sys.getsizeof(list_ver)

print("\n3) 제너레이터 vs 리스트 메모리 비교")
print(f"  제너레이터 : {gen_size} bytes")
print(f"  리스트     : {list_size} bytes")

# assert: generator sys.getsizeof < list 확인
assert gen_size < list_size, \
    f"메모리 비교 실패: gen({gen_size}B) >= list({list_size}B)"


# =======================================================================
# 실습 4) 종합 — 월별 카테고리 매출 집계
# =======================================================================

# defaultdict(int): (month, category) 복합 키로 총매출 집계
#   기본값 0 이므로 키 없어도 바로 += 가능
monthly_cat: defaultdict = defaultdict(int)
for row in sales:
    monthly_cat[(row["month"], row["category"])] += row["amount"]

# 딕셔너리 컴프리헨션으로 (month, category) 기준 오름차순 정렬
monthly_cat_total: Dict[tuple, int] = {
    key: total
    for key, total in sorted(monthly_cat.items())
}

print("\n4) 월별 카테고리 매출 집계:")
current_month = None
for (month, category), total in monthly_cat_total.items():
    if month != current_month:      # 월이 바뀔 때 구분 출력
        print(f"\n{month}")
        current_month = month
    print(f"{category}: {total:,}원")

# top3 금액 내림차순 정렬
top3_monthly = sorted(monthly_cat_total.items(), key=lambda x: -x[1])[:3]

print("\nTop 3")
for rank, ((month, cat), total) in enumerate(top3_monthly, 1):
    print(f"{rank}. {month} / {cat}: {total:,}원")

# assert: top3 금액 내림차순 정렬 정확 확인
assert all(
    top3_monthly[i][1] >= top3_monthly[i + 1][1]
    for i in range(len(top3_monthly) - 1)
), "top3 내림차순 정렬 오류"
