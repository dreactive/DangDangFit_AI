# api/diet.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from app.services.diet_service import DietPlanner, DataPaths

router = APIRouter(prefix="/diet", tags=["diet"])

# ---------- Request/Response Models ----------
class DietRequest(BaseModel):
    user_type: str = Field(..., description="공복 혈당이 높은 유형 / 식후 혈당이 높은 유형 / 체중 증가가 과도한 유형 / 인슐린 복용중인 형")
    preferences: List[str] = Field(default_factory=list, description="사용자 선호 음식 키워드")
    allergies: List[str] = Field(default_factory=list, description="사용자 알러지(차단) 키워드")

class DietResponse(BaseModel):
    plan: Dict[int, List[List[str]]]  # {0:[ [..아침..], [..점심..], [..저녁..] ], 1: ...}
    
    
class DietOneRequest(BaseModel):
    user_type: str = Field(..., description="공복 혈당이 높은 유형 / 식후 혈당이 높은 유형 / 체중 증가가 과도한 유형 / 인슐린 복용중인 형")
    preferences: List[str] = Field(default_factory=list, description="사용자 선호 음식 키워드")
    allergies: List[str] = Field(default_factory=list, description="사용자 알러지(차단) 키워드")
    oneDay: int = Field(..., description="선택한 요일")
    oneTime: int = Field(..., description="아침,점심,저녁,간식")

class DietOneResponse(BaseModel):
    plan: List[str]  

# ---------- Router State ----------
# 환경에 맞게 실제 경로 세팅
DATA_PATHS = DataPaths(
    recipes_dir="app/data/recipes/recipes",
    food_kor_path="app/data/diet/filtered_gdm_foods.xlsx",      # <- 실제 파일 경로로 바꿔주세요 (csv/xlsx 가능)
    food_usda_path=None,                             # 선택
    gi_kor_path="app/data/share/GI 지수 한국.pdf",                                # 선택(사전 추출본 권장)
    gi_intl_path="app/data/share/GI 지수 국제.pdf",                               # 선택(사전 추출본 권장)
    guidelines_dir="app/data/diet"                # 선택
)
planner = DietPlanner(DATA_PATHS)

# ---------- Endpoints ----------
@router.post("/weekly", response_model=DietResponse)
@router.post("/initial", response_model=DietResponse)
def generate_weekly_diet(req: DietRequest):
    """
    하루 3끼, 일주일치 식단(총 21식)을 생성해 반환.
    - 반환 포맷:
      {0:[["food3.json","food25.json","P101-101000100-0001"],
          [...점심...],
          [...저녁...]],
       1:[...],
       ...
      }
    - 각 끼니 항목은 레시피 파일명(예: food25.json) 또는 "식품코드" 값 문자열.
    """
    
    # Fasting Glucose High Type / Postprandial Glucose High Type / Excessive Weight Gain Type / Insulin Therapy Type
    valid_types = {"F", "P", "E", "I"} 
    if req.user_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"user_type must be one of {sorted(list(valid_types))}")

    plan = planner.generate_weekly_plan(
        user_type=req.user_type,
        preferences=req.preferences,
        allergies=req.allergies
    )
    return DietResponse(plan=plan)


@router.post("/one", response_model=DietOneResponse)
def generate_one_diet(req: DietOneRequest):
    temp = DietRequest(user_type=req.user_type, preferences=req.preferences,allergies=req.allergies)
    weekly = generate_weekly_diet(temp)
    return DietOneResponse(plan=weekly.plan[req.oneDay][req.oneTime])