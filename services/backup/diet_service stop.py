# services/diet_service.py
from __future__ import annotations
import os, json, random, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

import traceback
from fastapi import HTTPException

# -----------------------------
# 상단 상수 구간에 추가
# -----------------------------
GI_MAX_BY_SLOT = {
    "breakfast": 55,
    "lunch": 60,
    "dinner": 60,
    "dessert": 55,
}
GL_MAX_BY_SLOT = {
    "breakfast": 10,
    "lunch": 13,
    "dinner": 13,
    "dessert": 8,
}
SUGAR_MAX_BY_SLOT = {
    "breakfast": 8,
    "lunch": 12,
    "dinner": 12,
    "dessert": 10,
}

# 아침 절대 배제 키워드
BREAKFAST_BAN_KWS = ["크로와상","와플","머핀","도넛","소금빵","브레드","페이스트리","버터롤"]
# 고/저 GI 추정용
HIGH_GI_NAME_KWS = ["크로와상","도넛","와플","머핀","브레드","소금빵","케이크","페이스트리","버거","라면"]
LOW_GI_NAME_KWS  = ["현미","잡곡","귀리","보리","퀴노아","메밀","통밀","콩","두부","샐러드"]

# -----------------------------
# Config
# -----------------------------
@dataclass
class DataPaths:
    recipes_dir: str = "app/data/recipes/recipes"
    food_kor_path: Optional[str] = "app/data/share/음식 영양성분.xlsx"      # "음식 영양성분" (csv/xlsx) -> must have columns: 식품코드, 식품명, 에너지(kcal), 탄수화물(g), 단백질(g), 지방(g), 당류(g), 나트륨(mg), 식이섬유(g)
    food_usda_path: Optional[str] = None     # USDA FDC dump or subset (optional)
    gi_kor_path: Optional[str] = "app/data/share/GI 지수 한국.pdf"        # GI 지수 한국 (csv/xlsx or pre-extracted)
    gi_intl_path: Optional[str] = "app/data/share/GI 지수 국제.pdf"       # GI 지수 국제 (csv/xlsx or pre-extracted)
    guidelines_dir: Optional[str] = "app/data/diet"     # PDFs or extracted text; optional for RAG

# -----------------------------
# Lightweight “RAG” stub
# --- NEW: real RAG retriever ---
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class RAGDietKnowledge:
    def __init__(self, paths: DataPaths, index_dir: str = ".cache/rag_diet"):
        self.paths = paths
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vdb = self._build_or_load_index()

    def _load_docs(self):
        docs = []
        if self.paths.guidelines_dir and Path(self.paths.guidelines_dir).exists():
            # PDF 일괄 로드
            loader = DirectoryLoader(self.paths.guidelines_dir, glob="*.pdf", loader_cls=PyPDFLoader)
            docs = loader.load()
        # 필요하면 GI 테이블을 CSV로 추출해 텍스트로 합치세요.
        return docs

    def _build_or_load_index(self):
        idx_path = self.index_dir / "faiss_index"
        if (idx_path / "index.faiss").exists():
            return FAISS.load_local(
                str(idx_path),
                OpenAIEmbeddings(model="text-embedding-3-small"),
                allow_dangerous_deserialization=True
            )
        # 처음 한 번만 빌드
        docs = self._load_docs()
        if not docs:
            return None
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        vdb = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
        vdb.save_local(str(idx_path))
        return vdb

    def fetch_constraints(self, user_type: str) -> Dict[str, Any]:
        # 기본값
        constraints = {
            "prefer_low_gi": True,
            "meal_carb_g": (35, 55),
            "meal_kcal_max": 650,
            "sodium_mg_max": 1200,

            # ★ 슬롯별 기본 목표치 추가
            "breakfast_carb_g": (25, 40),
            "lunch_carb_g": (35, 60),
            "dinner_carb_g": (35, 60),
            "dessert_carb_g": (0, 30),

            # (선택) 저녁/점심 kcal 상한 살짝 여유
            "lunch_kcal_max": 700,
            "dinner_kcal_max": 700,
            
            # ★ 결측 배제 정책
            # any=True 이면 모든 영양 필드 중 하나라도 NaN이면 제외(매우 엄격)
            # any=False 이면 '주요 필드'에 NaN이 있으면 제외(권장)
            "exclude_if_nan_any": False,  
            "nan_critical_fields": ["carb_g", "kcal", "sugar_g", "sodium_mg"],
        }
        # RAG 질의(있을 때만)
        if self.vdb:
            print("RAG 작동중")
            q = f"임신성 당뇨 {user_type} 에 적합한 한 끼 탄수화물 범위, GI, 당류, 나트륨 제한, 아침/점심/저녁 차이 지침"
            hits = self.vdb.similarity_search(q, k=6)
            # ↓ 간단 파서(키워드 기반). 추후 프롬프트로 LLM에 요약/파싱 맡겨도 됨.
            text = "\n".join([h.page_content for h in hits]).lower()
            if "저gi" in text or "low gi" in text:
                constraints["prefer_low_gi"] = True
            if "당류" in text or "sugar" in text:
                constraints["max_added_sugar_g"] = 5
            if "나트륨" in text or "sodium" in text:
                constraints["sodium_mg_max"] = 1000
            if "아침" in text and ("탄수" in text or "carb" in text):
                constraints["breakfast_carb_g"] = (25, 40)

        # 케이스별 추가 튜닝(가이드라인 + 경험 규칙)
        if user_type == "공복 혈당이 높은 유형":
            constraints.setdefault("breakfast_carb_g", (25, 40))
            constraints["prefer_early_fiber"] = True
        elif user_type == "식후 혈당이 높은 유형":
            constraints["max_added_sugar_g"] = min(constraints["max_added_sugar_g"], 5)
            constraints["prefer_low_gi"] = True
        elif user_type == "체중 증가가 과도한 유형":
            constraints["meal_kcal_max"] = 550
            constraints["meal_carb_g"] = (30, 50)
        elif user_type == "인슐린 복용중인 형":
            constraints["prefer_low_gi"] = True

        return constraints


# -----------------------------
class DietKnowledge:
    """
    Extremely lightweight retrieval stub.
    In production, replace with your FAISS/Chroma + embeddings retriever
    loaded with:
      - 케이스별 분류.pdf
      - 영양 균형 기준.pdf
      - 재료 유형별 영향.pdf
      - GI 지수(한국/국제) 정리 테이블
    """
    def __init__(self, paths: DataPaths):
        self.paths = paths
        # You can preload extracted texts from PDFs (once) into self.notes
        self.notes: Dict[str, str] = {
            "case_rules": "공복↑: 아침 탄수 30~40g, 저GI/식이섬유↑ / 식후↑: 전체 저GI·당류↓, 단순당 회피 / 체중↑: kcal 목표 하향 / 인슐린: 균형 유지, 저GI·규칙성",
            "balance": "임당 임산부 균형: 탄수 40~50% 내 저GI 중심, 단백질 적정(1.1g/kg/day 근사), 불포화지방, 나트륨 과다 회피",
            "ingredients": "정제곡↓, 통곡/잡곡/콩류/채소↑, 과당·액상과당 음료 회피, 튀김/당 과다 간식 회피",
        }

    def fetch_constraints(self, user_type: str) -> Dict[str, Any]:
        """
        Returns numeric/boolean constraints derived from the domain knowledge.
        Customize per your guidelines.
        """
        # Defaults
        constraints = {
            "prefer_low_gi": True,
            "meal_carb_g": (35, 55),
            "meal_kcal_max": 650,
            "sodium_mg_max": 1200,

            "breakfast_carb_g": (25, 40),
            "lunch_carb_g": (35, 60),
            "dinner_carb_g": (35, 60),
            "dessert_carb_g": (0, 30),
            "lunch_kcal_max": 700,
            "dinner_kcal_max": 700,

            # ★ 결측 배제 정책
            # any=True 이면 모든 영양 필드 중 하나라도 NaN이면 제외(매우 엄격)
            # any=False 이면 '주요 필드'에 NaN이 있으면 제외(권장)
            "exclude_if_nan_any": False,  
            "nan_critical_fields": ["carb_g", "kcal", "sugar_g", "sodium_mg"],
        }
        # Case-specific tweaks
        if user_type == "공복 혈당이 높은 유형":
            # Control breakfast carbs & GI tighter
            constraints["breakfast_carb_g"] = (25, 40)
            constraints["prefer_early_fiber"] = True
        elif user_type == "식후 혈당이 높은 유형":
            constraints["prefer_low_gi"] = True
            constraints["max_added_sugar_g"] = 5
        elif user_type == "체중 증가가 과도한 유형":
            constraints["meal_kcal_max"] = 550
            constraints["meal_carb_g"] = (30, 50)
        elif user_type == "인슐린 복용중인 형":
            constraints["prefer_low_gi"] = True
            constraints["meal_carb_g"] = (35, 55)
        return constraints

# -----------------------------
# Loaders
# -----------------------------
def _safe_read_table(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    ext = Path(path).suffix.lower()
    try:
        if ext in [".csv", ".tsv"]:
            sep = "\t" if ext == ".tsv" else ","
            return pd.read_csv(path, sep=sep)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        elif ext in [".parquet"]:
            return pd.read_parquet(path)
    except Exception:
        return None
    return None

def load_recipes(recipes_dir: str) -> List[Dict[str, Any]]:
    items = []
    p = Path(recipes_dir)
    if not p.exists():
        return items
    for fp in sorted(p.glob("food*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_file"] = fp.name  # keep file name for return
            # Optional normalization
            items.append(data)
        except Exception as e:
            print(traceback.format_exc())
            continue
    return items

# -----------------------------
# 전처리리
# -----------------------------
import math

def _nz(x, default=0.0):
    """None/NaN/이상치 → default(기본 0.0)로 보정"""
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


# -----------------------------
# Nutrient & GI helpers
# -----------------------------
@dataclass
class FoodRow:
    code: str
    name: str
    kcal: float
    carb_g: float
    protein_g: float
    fat_g: float
    sugar_g: float
    sodium_mg: float
    fiber_g: float
    gi: Optional[float] = None

def _col(df: pd.DataFrame, key: str) -> Optional[str]:
    # Try common variations quickly
    candidates = [key, key.lower(), key.upper()]
    for c in candidates:
        if c in df.columns:
            return c
    # loose matching
    for c in df.columns:
        if key.replace(" ", "") in c.replace(" ", ""):
            return c
    return None

def build_food_index(df_kor: Optional[pd.DataFrame], df_usda: Optional[pd.DataFrame],
                     gi_index: Optional[pd.DataFrame]) -> Dict[str, FoodRow]:
    idx: Dict[str, FoodRow] = {}

    def add_row(row: Dict[str, Any], code_key: str, name_key: str,
                kcal_key: str, carb_key: str, protein_key: str, fat_key: str,
                sugar_key: str, sodium_key: str, fiber_key: str):
        try:
            code = str(row[code_key])
            idx[code] = FoodRow(
                code=code,
                name=str(row.get(name_key, "")),
                kcal=float(row.get(kcal_key, 0) or 0),
                carb_g=float(row.get(carb_key, 0) or 0),
                protein_g=float(row.get(protein_key, 0) or 0),
                fat_g=float(row.get(fat_key, 0) or 0),
                sugar_g=float(row.get(sugar_key, 0) or 0),
                sodium_mg=float(row.get(sodium_key, 0) or 0),
                fiber_g=float(row.get(fiber_key, 0) or 0),
                gi=None
            )
        except Exception:
            pass

    # Korean DB
    if df_kor is not None and len(df_kor) > 0:
        ck = _col(df_kor, "식품코드") or _col(df_kor, "code") or df_kor.columns[0]
        nk = _col(df_kor, "식품명") or _col(df_kor, "name") or df_kor.columns[1]
        kcal = _col(df_kor, "에너지(kcal)") or _col(df_kor, "kcal")
        carb = _col(df_kor, "탄수화물(g)") or _col(df_kor, "carb")
        protein = _col(df_kor, "단백질(g)") or _col(df_kor, "protein")
        fat = _col(df_kor, "지방(g)") or _col(df_kor, "fat")
        sugar = _col(df_kor, "당류(g)") or _col(df_kor, "sugar")
        sodium = _col(df_kor, "나트륨(mg)") or _col(df_kor, "sodium")
        fiber = _col(df_kor, "식이섬유(g)") or _col(df_kor, "fiber")
        for _, r in df_kor.iterrows():
            add_row(r, ck, nk, kcal, carb, protein, fat, sugar, sodium, fiber)

    # USDA (optional, merge by code-like key if you maintain one; else skip or use FDC_ID)
    # Skipped here unless your schema aligns

    # GI merge (optional): expect columns like ['식품명','GI'] or ['code','GI']
    if gi_index is not None and len(gi_index) > 0:
        gi_name = _col(gi_index, "식품명") or _col(gi_index, "name")
        gi_code = _col(gi_index, "식품코드") or _col(gi_index, "code")
        gi_val = _col(gi_index, "GI")
        if gi_val:
            for _, r in gi_index.iterrows():
                gi_value = r.get(gi_val, None)
                if gi_value is None or (isinstance(gi_value, float) and math.isnan(gi_value)):
                    continue
                if gi_code and str(r.get(gi_code, "")) in idx:
                    idx[str(r.get(gi_code))].gi = float(gi_value)
                elif gi_name:
                    # loose name join
                    name = str(r.get(gi_name, "")).strip()
                    for k, v in idx.items():
                        if v.name and name in v.name:
                            v.gi = float(gi_value)

    return idx

# 슬롯별 목표치 헬퍼
def _slot_targets(constraints: dict, slot: str):
    """슬롯별 (carb_lo, carb_hi), kcal_max, sodium_max 반환"""
    carb = constraints.get(f"{slot}_carb_g", constraints.get("meal_carb_g", (35, 55)))
    kcal_max = constraints.get(f"{slot}_kcal_max", constraints.get("meal_kcal_max", 650))
    sodium_max = constraints.get(f"{slot}_sodium_mg_max", constraints.get("sodium_mg_max", 1200))
    return carb, kcal_max, sodium_max

import math

def _is_missing(x) -> bool:
    try:
        if x is None:
            return True
        if isinstance(x, float) and math.isnan(x):
            return True
        return False
    except Exception:
        return True

def _as_inf_if_missing(x) -> float:
    """결측이면 +inf 반환 → 임계 비교에서 자동 탈락"""
    try:
        if _is_missing(x):
            return float("inf")
        return float(x)
    except Exception:
        return float("inf")

# GI 추론 + GL 계산 추가

def infer_gi_from_name(name: str) -> float:
    n = (name or "").lower()
    if any(k in n for k in (kw.lower() for kw in HIGH_GI_NAME_KWS)):
        return 70.0
    if any(k in n for k in (kw.lower() for kw in LOW_GI_NAME_KWS)):
        return 45.0
    return 60.0  # 모르면 보수적으로 중간~높음

def ensure_gi(value: Optional[float], name: str) -> float:
    """GI가 None이면 이름으로 추정."""
    return float(value) if _finite(value) else infer_gi_from_name(name)

def gl_of(gi: float, carb_g: float) -> float:
    return (gi * max(0.0, carb_g)) / 100.0

# -----------------------------
# 소프트 방식으로 중복
# -----------------------------
import math, random
from collections import deque, defaultdict

class PenaltyMemory:
    def __init__(self, maxlen=18, base_penalty=0.80, decay=0.85):
        self.buf = deque(maxlen=maxlen)  # 최근 선택된 아이템 id들
        self.freq = defaultdict(int)
        self.base_penalty = base_penalty
        self.decay = decay

    def penalty(self, item_id: str) -> float:
        # 최근 많이/자주 뽑힌 아이템일수록 패널티 ↑ (0~1)
        k = self.freq.get(item_id, 0)
        if k <= 0:
            return 1.0
        # k번 등장 시 base_penalty^k를 주고, 버퍼가 가득찰수록 decay도 추가
        return (self.base_penalty ** k) * (self.decay ** (len(self.buf) // 6))

    def push(self, item_id: str):
        self.buf.append(item_id)
        self.freq[item_id] += 1
        # 버퍼 초과 제거 시 freq 감소
        while len(self.buf) > self.buf.maxlen:
            old = self.buf.popleft()
            self.freq[old] -= 1
            if self.freq[old] <= 0:
                del self.freq[old]

def _softmax_sample(scored_items, topn=6, temperature=0.7):
    # scored_items: List[Tuple[score(float), id(str)]], 높은 점수일수록 좋음
    if not scored_items:
        return None
    pool = scored_items[:topn]
    # 점수를 temperature softmax 확률로
    mx = max(s for s, _ in pool)
    exps = [math.exp((s - mx)/max(1e-6, temperature)) for s, _ in pool]
    Z = sum(exps)
    probs = [e/Z for e in exps]
    r = random.random()
    cum = 0.0
    for (s, it), p in zip(pool, probs):
        cum += p
        if r <= cum:
            return it
    return pool[-1][1]

# -----------------------------
# Candidate generator
# -----------------------------
def filter_recipe_by_allergy_and_pref(recipe: Dict[str, Any], allergies: List[str], prefs: List[str]) -> bool:
    text = json.dumps(recipe, ensure_ascii=False)
    # Allergies: hard block
    for a in allergies:
        if a and a.strip() and a.lower() in text.lower():
            return False
    # Preferences: soft prefer (handled in scoring)
    return True

def score_recipe_for_case(recipe: Dict[str, Any], constraints: Dict[str, Any], meal_slot: str) -> float:
    """
    Heuristic scoring:
    - prefer low sugar sauce mentions
    - prefer grains/beans/mushrooms/veggies
    - breakfast: slightly lower carbs
    """
    text = json.dumps(recipe, ensure_ascii=False).lower()
    s = 0.0
    for kw in ["현미", "잡곡", "콩", "버섯", "채소", "샐러드", "두부", "오트", "퀴노아", "보리"]:
        if kw in text:
            s += 1.0
    for bad in ["설탕", "시럽", "튀김", "액상과당", "과자", "케이크", "달콤"]:
        if bad in text:
            s -= 1.0
    if constraints.get("prefer_low_gi", True):
        for gi_kw in ["잡곡밥", "콩밥", "귀리", "메밀", "통밀", "보리"]:
            if gi_kw in text:
                s += 0.5
    if meal_slot == "breakfast" and "밥" in text:
        s -= 0.2  # keep breakfast carbs modest unless fiber-rich
        if any(x in text for x in ["잡곡", "콩밥", "현미"]):
            s += 0.4
    return s

def select_recipes(recipes: List[Dict[str, Any]], constraints: Dict[str, Any],
                  allergies: List[str], prefs: List[str], k: int,
                  meal_slot: str) -> List[str]:
    cand = []
    for r in recipes:
        if filter_recipe_by_allergy_and_pref(r, allergies, prefs):
            sc = score_recipe_for_case(r, constraints, meal_slot)
            # preference bonus
            if any(p.lower() in json.dumps(r, ensure_ascii=False).lower() for p in prefs):
                sc += 0.5
            cand.append((sc, r["_file"]))
    cand.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in cand[:k]]

def select_foodcodes(index: Dict[str, FoodRow], constraints: Dict[str, Any],
                     allergies: List[str], prefs: List[str], k: int,
                     meal_slot: str) -> List[str]:
    out = []
    (carb_range, kcal_max, sodium_max) = _slot_targets(constraints, meal_slot)
    lo, hi = carb_range

    strict_any = constraints.get("exclude_if_nan_any", False)
    critical = set(constraints.get("nan_critical_fields", ["carb_g","kcal","sugar_g","sodium_mg"]))

    def _has_forbidden_nan(row: FoodRow) -> bool:
        # 엄격 모드: 어떤 필드라도 NaN이면 제외
        if strict_any:
            fields = ["kcal","carb_g","protein_g","fat_g","sugar_g","sodium_mg","fiber_g"]
            return any(_is_missing(getattr(row, f, None)) for f in fields)
        # 권장 모드: 주요 필드에 NaN 있으면 제외
        return any(_is_missing(getattr(row, f, None)) for f in critical)

    gi_slot_max = GI_MAX_BY_SLOT.get(meal_slot, 60)
    gl_slot_max = GL_MAX_BY_SLOT.get(meal_slot, 12)
    sugar_slot_max = SUGAR_MAX_BY_SLOT.get(meal_slot, constraints.get("max_added_sugar_g", 10))

    for code, row in index.items():
        # 알레르기
        if any(a.strip() and a.lower() in row.name.lower() for a in allergies):
            continue


        # ★ 아침 절대 배제
        if meal_slot == "breakfast" and any(kw.lower() in row.name.lower() for kw in BREAKFAST_BAN_KWS):
            continue
          
        # ★ 결측 배제
        if _has_forbidden_nan(row):
            continue

        # GI/GL 계산 (서빙량 가정)
        sv = _slot_default_serving(meal_slot)
        gi_val = ensure_gi(row.gi, row.name)

        # 100g 기준 느슨한 탄수 창
        # carb = _as_inf_if_missing(row.carb_g)
        # if not (lo * 0.5 <= carb <= hi * 1.5):
        #     continue

        carb = _as_inf_if_missing(row.carb_g) * (sv / 100.0)
        kcal = _as_inf_if_missing(row.kcal) * (sv / 100.0)
        sugar = _as_inf_if_missing(row.sugar_g) * (sv / 100.0)
        sodium = _as_inf_if_missing(row.sodium_mg) * (sv / 100.0)
        
        if gi_val > gi_slot_max:
            continue
        if gl_of(gi_val, carb) > gl_slot_max:
            continue
        if kcal > kcal_max:
            continue
        if sugar > constraints.get("max_added_sugar_g", 10) + 5:
            continue
        if sodium > sodium_max:
            continue
          
        # 탄수 목표 대역(서빙 반영)
        if not (lo <= carb <= hi):
            continue
          
        # 이름 페널티
        name_lower = row.name.lower()
        name_penalty = 50 if any(kw in name_lower for kw in (k.lower() for k in HIGH_GI_NAME_KWS)) else 0


        pref_boost = 1 if any(p.lower() in row.name.lower() for p in prefs) else 0
        gi_for_score = (row.gi if _finite(row.gi) else 50.0)
        
        BAD_NAME_KWS = ["피자","도넛","케이크","쿠키","패스트푸드","버거","라면","튀김","프라이드","콜라","사이다","디저트"]
        BAD_NAME_PENALTY = 50  # 점수에서 빼기
        name_lower = row.name.lower()
        name_penalty = BAD_NAME_PENALTY if any(kw in name_lower for kw in BAD_NAME_KWS) else 0
        score = (1000 - gi_for_score) - name_penalty + pref_boost*10 + max(0, 10 - abs(((lo+hi)/2) - float(carb)))
        
        # score = (1000 - gi_for_score) + pref_boost*10 + max(0, 10 - abs(((lo+hi)/2) - float(carb)))
        out.append((score, code))

    out.sort(key=lambda x: x[0], reverse=True)

    # ★ 후보가 너무 적으면 자동 완화(옵션): any→critical 완화
    if len(out) < max(4, k//2) and strict_any:
        # 완화 재시도
        relaxed = constraints.copy()
        relaxed["exclude_if_nan_any"] = False
        return select_foodcodes(index, relaxed, allergies, prefs, k, meal_slot)

    return [code for _, code in out[:k]]



# ---------------------------------------- 
# Meal-level nutrition helpers (ADD)
# ----------------------------------------
def _nutrition_from_recipe(recipe: dict) -> dict:
    n = recipe.get("nutrition") or {}
    return {
        "kcal": _nz(n.get("kcal")),
        "carb_g": _nz(n.get("carb_g")),
        "protein_g": _nz(n.get("protein_g")),
        "fat_g": _nz(n.get("fat_g")),
        "sugar_g": _nz(n.get("sugar_g")),
        "sodium_mg": _nz(n.get("sodium_mg")),
        "fiber_g": _nz(n.get("fiber_g")),
        "gi": (float(n.get("gi")) if _finite(n.get("gi")) else None),
        "carb_weight": _nz(n.get("carb_g")),
    }


def _nutrition_from_foodrow(row: FoodRow, serving_g: float) -> dict:
    ratio = max(0.0, serving_g) / 100.0  # FoodRow는 보통 100g 기준
    return {
        "kcal": _nz(row.kcal) * ratio,
        "carb_g": _nz(row.carb_g) * ratio,
        "protein_g": _nz(row.protein_g) * ratio,
        "fat_g": _nz(row.fat_g) * ratio,
        "sugar_g": _nz(row.sugar_g) * ratio,
        "sodium_mg": _nz(row.sodium_mg) * ratio,
        "fiber_g": _nz(row.fiber_g) * ratio,
        "gi": (float(row.gi) if _finite(row.gi) else None),
        "carb_weight": _nz(row.carb_g) * ratio,
    }


def _slot_default_serving(slot: str) -> float:
    # 슬롯별 기본 섭취량 가정(보수적)
    if slot == "breakfast": return 140.0
    if slot == "lunch": return 170.0
    if slot == "dinner": return 170.0
    if slot == "dessert": return 80.0
    return 150.0

def _sum_meal_nutrition(items: List[dict]) -> dict:
    total = {k: 0.0 for k in [
        "kcal","carb_g","protein_g","fat_g","sugar_g","sodium_mg","fiber_g","carb_weight"
    ]}
    gis = []
    for n in items:
        for k in total.keys():
            total[k] += _nz(n.get(k))
        gi_val = n.get("gi", None)
        if _finite(gi_val):
            gis.append((float(gi_val), _nz(n.get("carb_weight"))))
    if gis:
        wsum = sum(w for _, w in gis)
        total["gi_weighted"] = (sum(gi*w for gi, w in gis) / wsum) if wsum > 0 else (sum(gi for gi, _ in gis) / len(gis))
    else:
        total["gi_weighted"] = None
    return total

def _rule_check_meal(total: dict, constraints: dict, meal_slot: str) -> Tuple[bool, Dict[str, Any]]:
    (carb_range, kcal_max, sodium_max) = _slot_targets(constraints, meal_slot)
    lo, hi = carb_range
    
    gi_slot_max = GI_MAX_BY_SLOT.get(meal_slot, 60)
    gl_slot_max = GL_MAX_BY_SLOT.get(meal_slot, 12)
    sugar_slot_max = SUGAR_MAX_BY_SLOT.get(meal_slot, constraints.get("max_added_sugar_g", 10))


    ok = True
    report = {}

    def flag(key, cond, msg):
        nonlocal ok
        if cond is False:
            ok = False
            report[key] = msg

    def chk(key, cond, msg):
        val = total.get(key)
        if _finite(val):
            flag(key, cond, msg)
        else:
            report[f"{key}_warn"] = "값 없음(무시)"

    chk("carb_g", lo <= total["carb_g"] <= hi, f"탄수 {total['carb_g']:.1f}g (목표 {lo}~{hi})")
    chk("kcal", total["kcal"] <= kcal_max, f"kcal {total['kcal']:.0f} > max")
    chk("sodium_mg", total["sodium_mg"] <= sodium_max, f"나트륨 {total['sodium_mg']:.0f} > max")
    chk("sugar_g", total["sugar_g"] <= sugar_slot_max, f"당류 {total['sugar_g']:.1f} > max")

    # GI/GL (끼니 가중 GI도 보고, 가중 GL이 더 실용적이면 계산)
    giw = total.get("gi_weighted")
    if _finite(giw):
        flag("gi", giw <= gi_slot_max, f"가중 GI {giw:.0f} > {gi_slot_max}")
    else:
        report["gi_warn"] = "GI 없음(무시)"

    # 총 GL(간단 근사: 가중 GI × 총 탄수 / 100)
    if _finite(giw) and _finite(total.get("carb_g")):
        gl_total = gl_of(giw, total["carb_g"])
        flag("gl", gl_total <= gl_slot_max, f"총 GL {gl_total:.1f} > {gl_slot_max}")
        report["gl_total"] = round(gl_total, 1)
    else:
        report["gl_warn"] = "GL 계산불가"

    return ok, report


# ---------- RAG context + (optional) LLM check (ADD) ----------
def _rag_snippets(knowledge: RAGDietKnowledge, user_type: str) -> str:
    if getattr(knowledge, "vdb", None) is None:
        return ""
    q = f"임신성 당뇨 {user_type} 한 끼 권장: GI, 탄수, 당류, 나트륨, 아침/점심/저녁/간식 차이"
    hits = knowledge.vdb.similarity_search(q, k=4)
    return "\n\n".join([h.page_content[:1200] for h in hits])

def _llm_check_meal(meal_items: List[dict], total: dict, constraints: dict, user_type: str, rag_context: str) -> Optional[dict]:
    """
    LLM 연결 시 활성화. 연결이 없으면 None 반환 → 규칙 통과만으로 승인.
    - 기대 반환: {"pass": bool, "reasons": [...], "suggestions":[...]}
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 필요 시 모델 교체
        sys = (
            "너는 산모 영양 가이드라인을 준수하는 영양사다. "
            "입력된 끼니의 영양 합계가 제약을 만족하는지 검토하고, 부족/초과시 교체 제안도 해라. "
            "반드시 JSON으로만 답하라. keys=['pass','reasons','suggestions']"
        )
        user = {
            "constraints": constraints, "user_type": user_type,
            "meal_total": total, "items": meal_items,
            "guidelines_context": rag_context[:4000]
        }
        prompt = f"{sys}\n\nINPUT:\n{json.dumps(user, ensure_ascii=False)}"
        resp = llm([HumanMessage(content=prompt)])
        text = resp.content
        print(text)
        # json 파일 응답 형태가 정확한 json형태가 아니라서 발생한 오류
        print("0")
        # 1) 양쪽 따옴표 제거
        text = text.strip("`").strip()
        print("1")
        # 2) 맨 앞에 오는 'json' 같은 태그 제거
        if text.lower().startswith("json"):
            text = text[4:].strip()   # "json" 글자 제거
        print("2")
        # 3) 혹시 코드블록 같은게 붙었으면 정리
        text = text.replace("```json", "").replace("```", "").strip()
        print("3")
        # 4) { ... } 영역만 추출
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            text = match.group()

        
        print(text)
        # text = None  # ← LLM 미연결 시 None 유지
        if not text:
            return None
        data = json.loads(text)
        return {
            "pass": bool(data.get("pass", False)),
            "reasons": data.get("reasons", []),
            "suggestions": data.get("suggestions", []),
        }
    except Exception as e:
        print(traceback.format_exc())
        return None


# -----------------------------
# Meal plan assembly
# -----------------------------
MEAL_SLOTS = ["breakfast", "lunch", "dinner", "dessert"]

class DietPlanner:
    def __init__(self, paths: DataPaths):
        self.paths = paths
        self.knowledge = RAGDietKnowledge(paths)

        # Load datasets (lazy-tolerant)
        self.recipes = load_recipes(paths.recipes_dir)
        self.df_kor = _safe_read_table(paths.food_kor_path)
        self.df_usda = _safe_read_table(paths.food_usda_path)
        self.df_gi_kor = _safe_read_table(paths.gi_kor_path)
        self.df_gi_intl = _safe_read_table(paths.gi_intl_path)

        # Merge GI tables if both present
        gi_merge = None
        if self.df_gi_kor is not None and self.df_gi_intl is not None:
            gi_merge = pd.concat([self.df_gi_kor, self.df_gi_intl], axis=0, ignore_index=True)
        else:
            gi_merge = self.df_gi_kor if self.df_gi_kor is not None else self.df_gi_intl

        self.food_index = build_food_index(self.df_kor, self.df_usda, gi_merge)
        
        
        self.rec_mem = PenaltyMemory(maxlen=24, base_penalty=0.80, decay=0.90)
        self.food_mem = PenaltyMemory(maxlen=24, base_penalty=0.85, decay=0.92)
        random.seed(42)
    
    def _ranked_with_penalty(self, items: List[str], mem: PenaltyMemory, is_recipe=True):
        ranked = []
        # 기존 select_*는 상위 k만 돌려주므로, 넉넉히 k=20 정도로 받아오자
        for i, it in enumerate(items):
            # 원 점수를 대체로 보존하려면 앞쪽에 있을수록 점수 높다고 보고 선형 점수 부여
            base = 1000 - i  # 간단한 proxy; 필요 시 select_*에서 실제 score를 반환하도록 확장 가능
            pen = mem.penalty(it)
            ranked.append((base * pen, it))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def _sample_bundle(self, ranked, mem, count, topn=6, temperature=0.7):
        picked = []
        pool = ranked[:]  # (score, id)
        for _ in range(count):
            pool.sort(key=lambda x: x[0], reverse=True)
            it = _softmax_sample(pool, topn=topn, temperature=temperature)
            if not it:
                break
            picked.append(it)
            mem.push(it)
            # 같은 항목은 다음 샘플에서 점수를 더 낮게(즉시 페널티 재계산)
            pool = [(s * mem.penalty(_id), _id) if _id == it else (s, _id) for (s, _id) in pool]
            # 또는 선택된 it를 풀에서 제거
            pool = [(s, _id) for (s, _id) in pool if _id != it]
            if not pool:
                break
        return picked

    # ---------- DietPlanner methods (ADD INSIDE CLASS 식단 강화화) ----------
    

    def _choose_combo(self, slot: str) -> Tuple[int, int]:
        """
        (레시피 수, 식품 수) 동적 조합. 슬롯별 가중 무작위.
        - 아침: 가볍게/섬유↑ 경향
        - 점심: 균형
        - 저녁: 단품+레시피 다양
        - 간식: 1개 중심
        """
        rng = random.random()
        if slot == "breakfast":
            # (2,1),(1,1),(1,2),(2,0) 가중
            if rng < 0.35: return (2,1)
            if rng < 0.60: return (1,1)
            if rng < 0.80: return (1,2)
            return (2,0)
        elif slot == "lunch":
            if rng < 0.30: return (2,1)
            if rng < 0.55: return (1,2)
            if rng < 0.75: return (3,0)
            if rng < 0.95: return (0,3)
            return (1,1)
        elif slot == "dinner":
            if rng < 0.30: return (1,2)
            if rng < 0.55: return (2,1)
            if rng < 0.75: return (1,1)
            if rng < 0.90: return (0,3)
            return (3,0)
        else:  # dessert
            if rng < 0.50: return (1,0)
            if rng < 0.85: return (0,1)
            return (1,1)

    def _materialize_items(self, r_ids: List[str], f_ids: List[str], slot: str) -> List[dict]:
        rec_map = {r["_file"]: r for r in self.recipes}
        out: List[dict] = []
        for rid in r_ids:
            r = rec_map.get(rid)
            if not r: 
                continue
            n = _nutrition_from_recipe(r)
            out.append({"type":"recipe","id":rid,"name": r.get("title") or r.get("name") or rid, "nutrition": n})
        sv = _slot_default_serving(slot)
        for code in f_ids:
            row = self.food_index.get(code)
            if not row:
                continue
            n = _nutrition_from_foodrow(row, serving_g=sv)
            out.append({"type":"food","id":code,"name": row.name,
            "serving_g": sv,                 # ★ 추가
            "nutrition": n})
        return out

    def _meal_ok(self, meal_items: List[dict], constraints: dict, meal_slot: str, user_type: str, use_llm: bool) -> Tuple[bool, Dict[str, Any]]:
        total = _sum_meal_nutrition([it["nutrition"] for it in meal_items])
        ok, report = _rule_check_meal(total, constraints, meal_slot)
        if not ok:
            return False, {"why": "rule_fail", "report": report}
        if use_llm:
            ctx = _rag_snippets(self.knowledge, user_type)
            judge = _llm_check_meal(
                meal_items=[{"name": it["name"], "nutrition": it["nutrition"]} for it in meal_items],
                total=total, constraints=constraints, user_type=user_type, rag_context=ctx
            )
            if judge is not None and not judge.get("pass", False):
                return False, {"why": "llm_fail", "report": judge}
        return True, {"why": "ok"}

    # ---------- DietPlanner methods (ADD ADD INSIDE CLASS 식단 로직 재구성성) ----------

    # 슬롯별 유효 후보 세트 만들기기 
    def _gen_slot_candidates(
        self,
        slot: str,
        constraints: dict,
        preferences: List[str],
        allergies: List[str],
        tries_per_combo: int = 6,
        max_sets: int = 8,
        use_llm: bool = False,  # 후보 생성 단계에선 기본 비활성 (속도/비용↓)
    ):
        """
        한 슬롯(아침/점심/저녁/간식)에 대해
        - (r_need, f_need) 조합을 여러 번 바꿔가며
        - 샘플링 → 규칙통과(_meal_ok use_llm=False) 한 '세트'를 수집.
        반환: List[{'ids': [...], 'items':[...], 'score': float}]
        """
        r_candidates = select_recipes(self.recipes, constraints, allergies, preferences, k=24, meal_slot=slot)
        f_candidates = select_foodcodes(self.food_index, constraints, allergies, preferences, k=24, meal_slot=slot)
        print(f"[{slot}] rCands={len(r_candidates)}, fCands={len(f_candidates)}")

        r_ranked = self._ranked_with_penalty(r_candidates, self.rec_mem, is_recipe=True)
        f_ranked = self._ranked_with_penalty(f_candidates, self.food_mem, is_recipe=False)
        print(f"[{slot}] r_ranked={len(r_ranked)}, f_ranked={len(f_ranked)}")

        seen_ids = set()
        results = []
        attempts = 0

        while len(results) < max_sets and attempts < max_sets * tries_per_combo:
            attempts += 1
            r_need, f_need = self._choose_combo(slot)
            
            # 레시피 영양이 0인 경우가 있어 탄수가 모자라면 무조건 탈락하게 되는데 음식에서 보충할 수 있도록 최소 값 설정.
            if slot in ("lunch", "dinner") and f_need == 0:
              f_need = 1

            r_pick = self._sample_bundle(r_ranked, self.rec_mem, count=r_need, topn=6, temperature=0.75) if r_need > 0 else []
            f_pick = self._sample_bundle(f_ranked, self.food_mem, count=f_need, topn=6, temperature=0.75) if f_need > 0 else []

            ids = tuple([*r_pick, *f_pick])
            if not ids or ids in seen_ids:
                continue

            items = self._materialize_items(r_pick, f_pick, slot)
            
            # ★ 추가: 서빙을 탄수 목표에 맞게 자동 보정
            items = self._rebalance_servings(items, constraints, slot)
            
            
            ok, _info = self._meal_ok(items, constraints, slot, user_type="(hidden)", use_llm=False)  # 후보 생성단계는 룰만
            if not ok:
                if attempts <= 3:
                    tot = _sum_meal_nutrition([it["nutrition"] for it in items])
                    print(f"[{slot}] reject carbs={tot['carb_g']:.1f}, kcal={tot['kcal']:.0f}, sugar={tot['sugar_g']:.1f}, sodium={tot['sodium_mg']:.0f}")
                continue

            # 선호 보너스(이름에 prefs 포함) + 다양성(최근 선택 페널티 역수)
            pref_bonus = 0.0
            joined_names = " ".join(x["name"].lower() for x in items)
            for p in preferences:
                if p and p.strip() and p.lower() in joined_names:
                    pref_bonus += 0.4

            # 간단 가중치: 총 kcal가 목표 상한 대비 얼마나 여유있는지(너무 낮아도/높아도 페널티)
            total = _sum_meal_nutrition([it["nutrition"] for it in items])
            kcal_max = constraints.get("meal_kcal_max", 650.0)
            kcal = total.get("kcal", 0.0)
            kcal_score = max(0.0, 1.0 - abs(kcal - 0.8 * kcal_max) / (0.8 * kcal_max))  # 0~1

            score = pref_bonus + kcal_score
            results.append({"ids": list(ids), "items": items, "score": score})
            seen_ids.add(ids)

        # 점수순 정렬
        results.sort(key=lambda x: x["score"], reverse=True)
        print(results)
        return results

    #슬롯 후보 사전 점검
    def _validate_slot2cands(self, slot2cands: dict):
        """어떤 슬롯이라도 후보가 비어있으면 즉시 에러로 끊어 상위에서 대응."""
        empties = [s for s, c in slot2cands.items() if not c]
        if empties:
            raise ValueError(f"No candidates for slots: {empties}")

    # 하루 조합
    def _combine_day_beam(self, slot2cands: Dict[str, List[dict]], beam_k: int = 6):
        # 필수 가드
        self._validate_slot2cands(slot2cands)

        slots = list(slot2cands.keys())
        # beam: list of tuples [(path, score)]
        # path: list[tuple(slot_name, idx_within_that_slot)]
        """
        슬롯별 후보 세트를 받아 하루 조합을 beam search로 선택.
        스코어 = 슬롯별 세트 score 합 + 다양성 보너스
        """
        beams = [([], 0.0)]  # (선택 리스트[('slot', cand_idx)], score)

        for slot in MEAL_SLOTS:
            new_beams = []
            cands = slot2cands.get(slot, [])
            if not cands:
                # 후보가 없으면 빈 세트 하나 넣고 진행 (폴백)
                cands = [{"ids": [], "items": [], "score": -1.0}]

            for path, score in beams:
                for ci, c in enumerate(cands[:beam_k]):  # 각 슬롯 상위 beam_k만 탐색폭 제한
                    # 다양성 보너스: 동일 id 재등장에 약한 페널티
                    dup_penalty = 0.0
                    prev_ids = self._extract_prev_ids(slot2cands, path)
                    overlap = len(set(prev_ids) & set(c["ids"]))
                    if overlap > 0:
                        dup_penalty -= 0.5 * overlap
                    new_beams.append((path + [(slot, ci)], score + c["score"] + dup_penalty))

            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_k]  # 상위 K만 유지

        best_path, best_score = beams[0]
        return best_path, best_score
        

    # 후속 연산에서 ids가 없을 때 계산부를 안전하게 넘어가게 하기
    def _extract_prev_ids(self, slot2cands: dict, path: list[tuple[str, int]]) -> list:
        """path에 들어있는 (slot, idx)를 안전하게 풀어서 ids를 수집한다."""
        prev_ids = []
        for s, idx in path:
            cands = slot2cands.get(s, [])
            if 0 <= idx < len(cands):
                prev_ids.extend(cands[idx].get("ids", []))
            # else: 잘못된 path 항목은 무시 (혹은 로그 경고)
        return prev_ids

    # 영양 성분 규제 완화
    # 서빙 자동 보정기 추가
    def _rebalance_servings(self, items: List[dict], constraints: dict, slot: str) -> List[dict]:
        (carb_range, _, _) = _slot_targets(constraints, slot)
        lo, hi = carb_range

        # 식품만 조정
        food_idx = [(i, it) for i, it in enumerate(items) if it.get("type") == "food"]
        if not food_idx:
            return items

        def total_carb(itms): 
            return sum(_nz(x["nutrition"].get("carb_g")) for x in itms)

        cur = total_carb(items)
        if lo <= cur <= hi:
            return items

        rates = []
        for idx, it in food_idx:
            sg = it.get("serving_g", _slot_default_serving(slot))
            cg = _nz(it["nutrition"].get("carb_g"))
            rate = cg / max(sg, 1e-6)
            rates.append((idx, rate, sg))

        target = max(min((lo+hi)/2, hi), lo)
        delta = target - cur
        sum_rate = sum(r for _, r, _ in rates) or 1e-6

        # 디저트는 0g까지 허용, 다른 끼니는 최소 40g
        MIN_G, MAX_G = (0.0, 300.0) if slot == "dessert" else (40.0, 300.0)

        for idx, rate, sg in rates:
            if rate <= 0:
                continue
            gram_change = (abs(delta) * (rate / sum_rate)) / rate
            new_sg = min(MAX_G, sg + gram_change) if delta > 0 else max(MIN_G, sg - gram_change)

            row = self.food_index.get(items[idx]["id"])
            items[idx]["serving_g"] = new_sg
            items[idx]["nutrition"] = _nutrition_from_foodrow(row, new_sg)

        return items



    def generate_weekly_plan(self, user_type: str, preferences: List[str], allergies: List[str], use_llm: bool = False) -> Dict[int, List[List[str]]]:
        """
        하루 단위로 아침·점심·저녁·간식 후보를 먼저 만들고,
        beam search로 하루 조합을 고른 뒤(규칙 통과 보장),
        최종적으로 필요한 경우에만 LLM을 하루당 1회 호출해 재확인(옵션).
        """
        constraints = self.knowledge.fetch_constraints(user_type)
        plan: Dict[int, List[List[str]]] = {}

        for day in range(7):
          # 1) 슬롯별 유효 후보 세트 수집
          slot2cands: Dict[str, List[dict]] = {}
          for slot in MEAL_SLOTS:
              slot2cands[slot] = self._gen_slot_candidates(
                  slot=slot,
                  constraints=constraints,
                  preferences=preferences,
                  allergies=allergies,
                  tries_per_combo=6,
                  max_sets=8,
                  use_llm=False  # 후보 생성은 규칙만으로 빠르게
              )

          # 2) 하루 조합 선택(beam search)
          try:
              path, score = self._combine_day_beam(slot2cands, beam_k=6)
          except ValueError as e:
              raise HTTPException(status_code=422, detail=str(e))

          # 3) (선택) 하루 최종 LLM 점검: 각 끼니 합격이지만 전체 맥락에서 조언 받기
          if use_llm:
              try:
                  # 하루 설명용 요약(간단)
                  daily_items = []
                  for slot, ci in path:
                      daily_items.append({
                          "slot": slot,
                          "items": [{"name": it["name"], "nutrition": it["nutrition"]} for it in slot2cands[slot][ci]["items"]]
                      })
                  # 끼니별 총합만 간단히 취합해서 LLM에 한 번 보내고, pass 누락·경고만 확인 (구체 구현은 네 _llm_check_meal 변형으로 가능)
                  # 여기서는 비용/속도를 위해 생략 또는 Warn만 로그로 남겨도 충분.
              except Exception:
                  pass

          # 4) 최종 ids만 꺼내서 API 응답 포맷으로 정리
          day_meals: List[List[str]] = []
          for slot in MEAL_SLOTS:
              # path에 이 슬롯이 없을 수는 거의 없지만, 안전하게 처리
              match = [ci for (s, ci) in path if s == slot]
              if match:
                  day_meals.append(slot2cands[slot][match[0]]["ids"])
              else:
                  # 폴백: 후보가 비었을 때 최소 1개라도
                  rc = select_recipes(self.recipes, constraints, allergies, preferences, k=1, meal_slot=slot)
                  fc = select_foodcodes(self.food_index, constraints, allergies, preferences, k=1, meal_slot=slot)
                  day_meals.append(rc or fc or [])
          plan[day] = day_meals

        return plan

