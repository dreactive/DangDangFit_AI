# api/chatbot.py
from __future__ import annotations
from fastapi import APIRouter
from . import svc, AssessIn

from starlette.responses import JSONResponse

import math
from typing import Any
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False
def to_jsonable(obj: Any):
    # 기본형
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    # float (NaN/inf 처리)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # numpy 스칼라/배열 처리
    if _HAS_NUMPY:
        if isinstance(obj, (np.floating, np.integer)):
            f = float(obj)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        if isinstance(obj, np.ndarray):
            return [to_jsonable(x) for x in obj.tolist()]

    # dict / list / tuple / set 재귀 처리
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]

    # Pydantic v2 모델
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump())

    # Pandas Series/DataFrame은 .to_dict() 등으로 먼저 변환해 주세요.
    return str(obj)

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/ask")
def chat_ask(payload: AssessIn):
    """
    자유로운 문장 입력:
    - 예) "나 지금 갈비천왕 먹어도 돼?"
    """
    
    result = svc.assess(
        message=payload.message,
        recent_glucose=payload.recent_glucose,
        portion_g=payload.portion_g,
    )
    return JSONResponse(content=to_jsonable(result), status_code=200)
