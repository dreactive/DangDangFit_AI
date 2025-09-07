from __future__ import annotations
import os, re, json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

import traceback

# RAG(유료모델) — OpenAI 계열/호환 엔드포인트
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from ..db.pg_retriever import PgVectorRetriever  # 새로 추가한 retriever

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv("ddf/.env"))   # .env 파일 읽어서 os.environ에 등록

# DATA_DIR = Path("/mnt/data")  # 업로드 경로 그대로 사용
INTAKE_DIR = Path("app/data/intake")  # 업로드 경로 그대로 사용
SHARE_DIR = Path("app/data/share")  # 업로드 경로 그대로 사용
PATHS = {
    "csv_score": INTAKE_DIR / "수치인 아웃 판단.csv",
    "xlsx_food": SHARE_DIR / "음식 영양성분.xlsx",
    "xlsx_pack": INTAKE_DIR / "가공식품 영양성분.xlsx",
    "xlsx_ingr": SHARE_DIR / "원재료 영양성분.xlsx",
    "pdf_gi_kr": SHARE_DIR / "GI 지수 한국.pdf",
    "pdf_gi_intl": SHARE_DIR / "GI 지수 국제.pdf",
    "pdf_guide1": INTAKE_DIR / "가이드라인1.pdf",
    "pdf_guide2": INTAKE_DIR / "가이드라인2.pdf",
    "pdf_guide3": INTAKE_DIR / "가이드라인3.pdf",
    "pdf_bal": INTAKE_DIR / "영양 균형 기준준.pdf",
}

# GI 카테고리(국제 테이블 기준: low ≤55, 56–69, ≥70) :contentReference[oaicite:0]{index=0}
GI_LOW_MAX = 55
GI_MED_MAX = 69

def _safe_read_excel(p: Path) -> Optional[pd.DataFrame]:
    try:
        if p.exists(): return pd.read_excel(p)
    except Exception: ...
    return None

def _safe_read_csv(p: Path) -> Optional[pd.DataFrame]:
    try:
        if p.exists(): return pd.read_csv(p)
    except Exception: ...
    return None

class HybridService:
    """
    - 룰(정량): 음식 영양(탄수), GI → GL 산출, 최근혈당으로 보수적 보정
      • GL = GI × 가용 탄수(g) / 100, DGI/DGL 개념은 한국 논문 수식 참고(설명 근거) :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}
    - RAG(유료 LLM): GI/식후 목표·설명 등 근거 스니펫 요약
    - 데이터: /mnt/data 의 업로드 파일을 서비스에서 직접 로드(초미니)
    """
    def __init__(self):
        # 1) 테이블 로드 (있으면)
        self.tbl_food  = _safe_read_excel(PATHS["xlsx_food"])
        self.tbl_pack  = _safe_read_excel(PATHS["xlsx_pack"])
        self.tbl_ingr  = _safe_read_excel(PATHS["xlsx_ingr"])
        self.tbl_score = _safe_read_csv(PATHS["csv_score"])  # 평균/최대 임계치가 들어있다면 활용
        # 2) RAG 빌드 (GI 한국/국제 PDF -> 벡터스토어)
        
        self.llm = None
        self.retriever = None
        try:
            # (임베딩 모델 이름이 필요하면 .env에서 가져와 retriever에 넘겨줌)
            embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

            # DB에서 바로 검색하는 retriever
            self.retriever = PgVectorRetriever(
                k=4,
                embed_model=embed_model,
                pg_host=os.getenv("PG_HOST"),
                pg_port=int(os.getenv("PG_PORT", "5432")),
                pg_db=os.getenv("PG_DB"),
                pg_user=os.getenv("PG_USER"),
                pg_password=os.getenv("PG_PASSWORD"),
                filter_doc_ids=["f4a25ee1-9f6d-4bfa-bc2c-6ae1ac865d11" # GI 지수 한국
                                ,"dfe0a3a8-2a6e-4b11-9f33-5e7c91afa6b2" # GI 지수 국제
                                ,"0b6bddd3-fd1b-40ac-8ab6-3e242d13907d" # 가이드라인 1
                                ,"71ca6519-640e-42c1-af45-a2f1762dcf9c" # 가이드라인 2
                                ,"dce0f29f-c0c7-4738-ab5d-3f0a21000396" # 가이드라인 3
                                ,"55938d52-427c-46cc-9c3e-767ad4cc68bb" # 영양균형 기준준
                                ]
            )

            self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.2,
            )
        except Exception as e:
            # RAG/LLM 비활성(키 미설정 등)이어도 룰 판단은 동작
            print("DB 안됨.")
            self.retriever = None
            self.llm = None
            # print(traceback.format_exc())

    # ----------------- Public API -----------------
    def assess(self, message: str, recent_glucose: Optional[List[int]] = None,
              portion_g: Optional[float] = None) -> Dict[str, Any]:
        food = self._guess_food(message)
        hit = self._lookup_nutrition(food)

        
        
        # 탄수화물 g
        carb_g = self._pick_number(hit["row"], ["탄수", "carb"]) if hit else None
        # 1회 제공량 g (요청 portion이 있으면 우선)
        serving_g = portion_g or self._pick_number(hit["row"], ["1회", "제공", "섭취", "중량", "serv", "portion"]) if hit else None

        
        # GI 값(직접값 우선, 없으면 휴리스틱)
        gi_val = self._pick_number(hit["row"], ["GI", "glycemic"]) if hit else None
        if gi_val is None:
            gi_val = self._heuristic_gi(food)

        
        # GL 계산 (탄수 정보 없으면 None)
        # gl = (gi_val * float(carb_g) / 100.0) if (carb_g is not None) else None
        carb_g_raw = self._pick_number(hit["row"], ["탄수", "carb"]) if hit else None
        serving_g  = portion_g or self._pick_number(hit["row"], ["1회","제공","섭취","중량","serv","portion"]) if hit else None

        
        carb_per_serv = self._infer_carb_per_serving((hit or {}).get("row"), serving_g)
        # GL = GI * (carb_g_per_serving) / 100
        gl = (gi_val * carb_per_serv / 100.0) if (gi_val is not None and carb_per_serv is not None) else None

        
        gl_level = self._classify_gl(gl)

        # 최근 혈당 보정(간단): 평균≥140 또는 최대≥180면 한 단계 상향
        final_level = self._adjust_by_recent(gl_level, recent_glucose)

        # 결과 요약
        decision_map = {"낮음": "가능", "중간": "주의", "높음": "비추천", "정보부족": "주의"}
        probability_map = {"낮음": 0.8, "중간": 0.5, "높음": 0.2, "정보부족": 0.5}
        decision = decision_map.get(final_level, "주의")
        prob = probability_map.get(final_level, 0.5)
        
        # 권장 분량(목표 GL≤10 기준 단순 비례)
        rec_portion = self._recommend_portion(serving_g, gl, target_gl=10.0)

        # RAG로 근거/설명 보강(있으면)
        rag_answer = self._rag_explain(message, food, gi_val, carb_g, gl, gl_level, final_level)
        
        return {
            "input": message,
            "matched_food": food,
            "nutrition_hit": bool(hit),
            "nutrition_source": (hit or {}).get("table"),
            "serving_g": serving_g,
            "carb_g": carb_g,
            "gi": gi_val,
            "gl": round(gl, 1) if gl is not None else None,
            "gl_level": gl_level,
            "adjusted_level": final_level,
            "decision": decision,
            "probability_ok": round(prob, 2),
            "recommended_portion_g": rec_portion,
            "explanation": rag_answer or self._fallback_explain(food, gi_val, gl, gl_level, final_level),
        }

    # ----------------- 내부 유틸 -----------------
    def _guess_food(self, msg: str) -> str:
        msg = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", msg)
        toks = [t for t in msg.split() if len(t) >= 2]
        if not toks: return msg.strip()
        # 브랜드/수식어 제거 힌트(가벼운 규칙)
        stop = {"지금","먹어도","돼","먹어","싶어","오늘","어제","배달","간식","점심","저녁"}
        cands = [t for t in toks if t not in stop]
        # 2~3개까지 결합 시도 (예: 교촌 허니콤보 → 두 토큰)
        if len(cands) >= 2:
            return " ".join(cands[:3])
        return cands[0]

    def _lookup_nutrition(self, food: str) -> Optional[Dict[str, Any]]:
        def find_first(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
            if df is None or df.empty: return None
            text_cols = [c for c in df.columns if df[c].dtype == "O"]
            for c in text_cols:
                hit = df[df[c].astype(str).str.contains(food, case=False, na=False)]
                if len(hit) > 0: return {"table": c if isinstance(c, str) else "unknown", "row": hit.iloc[0].to_dict()}
            return None
        for df in [self.tbl_food, self.tbl_pack, self.tbl_ingr]:
            res = find_first(df)
            if res: return res
        return None

    def _pick_number(self, row: Optional[Dict[str, Any]], keys: List[str]) -> Optional[float]:
        if not row: return None
        for k, v in row.items():
            k_str = str(k).lower()
            if any(key.lower() in k_str for key in keys):
                try:
                    return float(str(v).replace(",", "").strip())
                except Exception: ...
        return None

    def _heuristic_gi(self, food: str) -> int:
        f = food.lower()
        if any(w in f for w in ["현미", "잡곡", "콩", "두부", "요거트", "샐러드"]): return 50
        if any(w in f for w in ["국수", "빵", "떡", "라면", "면", "파스타", "밥"]): return 70
        if any(w in f for w in ["감자", "고구마"]): return 60
        if any(w in f for w in ["치킨", "갈비", "삼겹", "스테이크"]): return 55
        return 60

    def _classify_gl(self, gl: Optional[float]) -> str:
        # GL 범주(일반 관례): ≤10 낮음, 11–19 중간, ≥20 높음 (근거식/정의는 DGI/DGL 수식 문헌 참고) :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}
        if gl is None: return "정보부족"
        if gl <= 10: return "낮음"
        if gl <= 19: return "중간"
        return "높음"

    def _adjust_by_recent(self, level: str, recent: Optional[List[int]]) -> str:
        if not recent: return level
        avg = sum(recent) / len(recent)
        mx = max(recent)
        order = ["낮음", "중간", "높음"]
        if (avg >= 140) or (mx >= 180):
            return order[min(order.index(level) + 1, len(order) - 1)] if level in order else level
        return level

    def _recommend_portion(self, serving_g: Optional[float], gl: Optional[float], target_gl: float) -> Optional[float]:
        if serving_g is None or gl is None or gl == 0: return None
        ratio = min(1.0, target_gl / gl)
        return round(float(serving_g) * ratio, 1)

    def _rag_explain(self, message: str, food: str, gi: float, carb_g: Optional[float],
                      gl: Optional[float], gl_level: str, final_level: str) -> Optional[str]:
        # print(self.retriever)
        print(self.llm)
        if not (self.retriever and self.llm): return None
        ctx_docs = self.retriever.get_relevant_documents(message + f" {food} GI {gi} GL {gl}")
        ctx = "\n\n---\n\n".join([f"[{i+1}] {d.page_content[:1000]}" for i, d in enumerate(ctx_docs)])
        sys = (
          """
            너는 임신성 당뇨(GDM) 사용자를 위한 ‘혈당 코치’다.
            원칙:
            1) 결론→이유→실행팁 순서로 간결하게 한국어로 답한다.
            2) GL 정의: GL = GI × 탄수화물(g)/100. GL 구간: ≤10 낮음, 11–19 중간, ≥20 높음.
            3) GI 구간: ≤55 낮음, 56–69 중간, ≥70 높음.
            4) 결론은 권고여부만 간단히 말하고, 값이 없으면 “추정”임을 명시하고 보수적으로 권고한다.
            5) 실용 팁은 추가해야하는 구체적인 재료, 줄여야하는 구체적인 재료를 명시해준다.
            6) 법적 고지: "일반 정보 제공이며 의료행위가 아니다. 개인 치료는 의료진 상담이 우선이다."를 부드럽게 말해줘.
            7) GL/GI 수치는 "음식수치(기준치)"형식으로 간단하게 명시한다.
            톤: 실용·과잉 확신 금지. 불확실성은 명확히 표시.
            출력 형식: 불릿 3~6줄, 과도한 장문 금지.

          """
        )
        user = f"""
[질문] {message}

[판단요약]
- 음식: {food}
- GI: {gi} (추정가능)
- 탄수화물 g: {carb_g}
- GL: {gl}
- GL 등급: {gl_level} → 최종: {final_level}

[문헌컨텍스트(요약용)]
{ctx}

요청사항:
- 1) “결론/이유/실행팁”만 간결히.
- 2) 값이 비어있으면 “추정/정보부족” 표기.
- 3) 필요 시 분량 조절 팁 1가지 포함.
"""
        rsp = self.llm.invoke([{"role":"system","content":sys},{"role":"user","content":user}])
        print(2)
        return rsp.content

    def _fallback_explain(self, food: str, gi: float, gl: Optional[float], gl_level: str, final_level: str) -> str:
        lines = []
        
        lines.append(f"결론: {final_level} 권장(안전도: { {'낮음':'높음','중간':'보통','높음':'낮음','정보부족':'불확실'}.get(final_level,'보통') })")
        lines.append(f"- 이유: GI≈{gi}, GL={gl if gl is not None else '정보부족'} → GL등급 {gl_level}. 최근 혈당에 따라 보수적 보정.")
        lines.append(f"- 팁: 단백질/식이섬유 먼저, 탄수화물·단맛 소스는 절제. 분량은 GL≤10 맞춰 조절.")
        # GI 카테고리/GL 정의 근거 간단 주석: 국제 GI 범주 & GL 정의(수식·범주) 참고. :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
        return "\n".join(lines)
      
      
    def _infer_carb_per_serving(self, row: Optional[Dict[str, Any]], serving_g: Optional[float]) -> Optional[float]:
        if not row: return None
        # 키 이름으로 힌트
        def pick(keys: List[str]) -> Optional[float]:
            for k, v in row.items():
                k_str = str(k).lower()
                if any(key in k_str for key in keys):
                    try: return float(str(v).replace(",", "").strip())
                    except: ...
            return None

        carb_val = pick(["탄수", "carb"])
        if carb_val is None:
            return None

        name_hit_100g = any(("100g" in str(k).lower() or "per 100" in str(k).lower()) for k in row.keys())
        name_hit_serv  = any(("1회" in str(k) or "serv" in str(k).lower() or "portion" in str(k).lower()) for k in row.keys())

        if name_hit_100g and serving_g:
            return round(carb_val * float(serving_g) / 100.0, 1)
        if name_hit_serv:
            return float(carb_val)
        # 정보 부족: serving_g와 결합 추정 불가
        return carb_val if serving_g is None else None