# router_recipe.py
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.services.recipe_service import transform_recipe_from_dict
from app.services.crawler import crawl_recipe_stub  # 위에서 분리한 파일

router = APIRouter(prefix="/recipe", tags=["recipe"])

class ConvertRequest(BaseModel):
    url: Optional[str] = Field(None, description="만개의레시피 URL")
    recipe_json: Optional[Dict[str, Any]] = Field(None, description="이미 크롤된 JSON")
    user_type: str = Field(..., description="FPG_HIGH | PPG_HIGH | WEIGHT_GAIN | INSULIN")
    allergies: List[str] = Field(default_factory=list)

# ✅ 동시 크롤 제한 (간단 세마포어)
CRAWL_CONCURRENCY = 2
_sema = asyncio.Semaphore(CRAWL_CONCURRENCY)

# ✅ 전체 작업 타임아웃 (크롤 + 변환)
TOTAL_TIMEOUT = 40  # 초

@router.post("/convert_recipe")
async def convert_recipe(req: ConvertRequest):
    if not req.url and not req.recipe_json:
        raise HTTPException(status_code=400, detail="url 또는 recipe_json 중 하나는 필요합니다.")

    async def _work():
        if req.recipe_json:
            data = req.recipe_json
        else:
            # 블로킹 크롤링을 워커 스레드로
            async with _sema:  # 동시성 제한
                data = await asyncio.to_thread(crawl_recipe_stub, req.url)
        return transform_recipe_from_dict(data, req.user_type, req.allergies)

    try:
        result = await asyncio.wait_for(_work(), timeout=TOTAL_TIMEOUT)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="크롤링/변환 타임아웃")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"변환 실패: {e}")
