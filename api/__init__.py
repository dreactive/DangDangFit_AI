# api/__init__.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

# 서비스 싱글턴 (RAG 인덱스/임베딩도 여기서 1회 초기화)
from app.services.hybrid_service import HybridService
svc = HybridService()

# 공용 요청 스키마 (chatboat/diet에서 재사용)
class AssessIn(BaseModel):
    message: str
    recent_glucose: Optional[List[Optional[int]]] = None
    portion_g: Optional[float] = None
    survey: Optional[Dict[str, Any]] = None  # (현재 서비스에서는 미사용)
