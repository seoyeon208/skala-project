"""
파일명: reporting.py
설명: 파이프라인 실행 중 콘솔 진행 상황 출력을 담당하는 Reporter 클래스.
기능:
    - Reporter.log(): 콘솔에 한 줄 출력한다.
    - Reporter.section(): 단계 구분용 헤더를 출력한다.
"""

from __future__ import annotations


class Reporter:
    """분석 진행 상황을 콘솔에 출력한다.

    보고서 내용은 각 단계가 반환하는 결과 구조체에서 나오므로, 여기 출력되는 내용은
    실행 중 진행 상황을 사람이 눈으로 확인하기 위한 용도다.
    """

    def log(self, message: str = "") -> None:
        """콘솔에 한 줄 출력한다."""
        print(message)

    def section(self, title: str) -> None:
        """단계 구분용 헤더를 출력한다."""
        self.log(f"\n===== {title} =====")
