from openai import OpenAI
import json
import os

import streamlit as st


def _get_openai_api_key():
    """Prefer Streamlit secrets in deployment, with env fallback for local runs."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    return os.getenv("OPENAI_API_KEY")


def generate_anomaly_report(sensor_data):
    api_key = _get_openai_api_key()
    if not api_key:
        return {
            "error": (
                "OPENAI_API_KEY is not configured. "
                "Add it to .streamlit/secrets.toml locally or Streamlit app secrets in deployment."
            )
        }

    try:
        client = OpenAI(api_key=api_key)

        system_prompt = """
        당신은 도금 및 크로메이트 표면처리 공정의 QA/QC 전문가이자 EHS 관리자입니다.
        당신의 임무는 실시간 센서 이상 데이터를 분석하여, 물리/화학적 원리(pH, 전압/전류, 온도, 반응 농도)에 기반한 정확한 원인 진단과 즉시 실행 가능한 SOP를 작성하는 것입니다.

        [작성 규칙]
        1. 근본 원인(Root Cause): 반드시 "가장 유력한 실제 공정 원인 1개" + "센서 오류 가능성 1개"로 분리하여 진단하십시오.
        2. 조치 지시의 구조: 즉각 조치는 반드시 "안전 정보(PPE) -> 물리적 행동 -> 대기 시간" 순서로 작성하고, 후속 단계는 "재측정 목표값 -> 실패 시 2차 보고(Escalation)"로 명확히 나누어 작성하십시오.
        3. 화학품 투입 기준: 절대량(ml, L) 대신 비율(%) 또는 사전 정의된 단계 기준으로 표현하십시오.
        4. 공정 변수별 대응:
           - pH: 조절제 주입 및 순환 펌프 가동
           - 온도: 냉각기/히터 출력 제어 및 열교환 밸브 확인
           - 전압/전류: 정류기 설정값 조정 및 접점 확인
        """

        user_prompt = f"""
        다음 공정 데이터를 분석하여 JSON 보고서를 작성하십시오.

        [현재 공정 상태 요약]
        - 대상 공정: {sensor_data.get('process_name', '크로메이트(Chromate)')}
        - 설비 용량: {sensor_data.get('tank_volume', 1500)}L
        - 발생 알람: {sensor_data.get('alarm_type', '해당 없음')}
        - 측정 수치: {sensor_data.get('current_value')} {sensor_data.get('unit')}
        - 정상 범위: {sensor_data.get('low_limit')} ~ {sensor_data.get('high_limit')} {sensor_data.get('unit')}
        - 해당 변수 추세: {sensor_data.get('trend', '데이터 없음')}
        - 타 변수 현재 상태: {sensor_data.get('other_sensors_status', '데이터 없음')}
        - 최근 30분간 조치 이력: {sensor_data.get('recent_actions', '없음')}
        """

        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "anomaly_report",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "AI_Report": {
                                "type": "object",
                                "properties": {
                                    "Issue": {"type": "string"},
                                    "Root_Cause": {
                                        "type": "object",
                                        "properties": {
                                            "Process_Cause": {"type": "string"},
                                            "Sensor_Cause": {"type": "string"},
                                        },
                                        "required": ["Process_Cause", "Sensor_Cause"],
                                        "additionalProperties": False,
                                    },
                                    "Corrective_Action": {"type": "string"},
                                    "Verification_Escalation": {"type": "string"},
                                    "Preventive_Action": {"type": "string"},
                                    "Confidence_Score": {"type": "integer"},
                                },
                                "required": [
                                    "Issue",
                                    "Root_Cause",
                                    "Corrective_Action",
                                    "Verification_Escalation",
                                    "Preventive_Action",
                                    "Confidence_Score",
                                ],
                                "additionalProperties": False,
                            }
                        },
                        "required": ["AI_Report"],
                        "additionalProperties": False,
                    },
                }
            },
        )

        if not getattr(response, "output_text", None):
            return {"error": f"AI response text is empty. raw response={response}"}

        parsed = json.loads(response.output_text)

        if "AI_Report" not in parsed:
            return {"error": f"AI response format is invalid: {parsed}"}

        return parsed

    except json.JSONDecodeError as exc:
        return {"error": f"Failed to parse AI response as JSON: {exc}"}
    except Exception as exc:
        return {"error": f"OpenAI API call failed: {exc}"}
