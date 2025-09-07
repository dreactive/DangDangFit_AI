# services/diet_service.py
from __future__ import annotations
import os, json, random, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

import traceback

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
            "max_added_sugar_g": 10,
            "prefer_low_gi": True,
            "meal_carb_g": (35, 55),
            "meal_kcal_max": 650,
            "sodium_mg_max": 1200,
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
            "max_added_sugar_g": 10,            # per meal
            "prefer_low_gi": True,
            "meal_carb_g": (35, 55),            # per meal, rough window
            "meal_kcal_max": 650,
            "sodium_mg_max": 1200,
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
    # filter: GI, kcal, sugar, sodium, carbs window
    out = []
    for code, row in index.items():
        # allergy filter (name-based)
        if any(a.strip() and a.lower() in row.name.lower() for a in allergies):
            continue
        # GI preference
        if constraints.get("prefer_low_gi", True) and row.gi is not None and row.gi > 55:
            continue
        # carbs window
        lo, hi = constraints.get("meal_carb_g", (30, 60))
        if meal_slot == "breakfast" and "breakfast_carb_g" in constraints:
            lo, hi = constraints["breakfast_carb_g"]
        if not (lo * 0.5 <= row.carb_g <= hi * 1.5):
            # single item grams are per 100g usually; keep a loose window
            continue
        # limits
        if row.kcal > constraints.get("meal_kcal_max", 650):
            continue
        if row.sugar_g > constraints.get("max_added_sugar_g", 10) + 5:  # crude bound (total sugar ~ added sugar proxy)
            continue
        if row.sodium_mg > constraints.get("sodium_mg_max", 1200):
            continue
        # prefer names with prefs
        pref_boost = 1 if any(p.lower() in row.name.lower() for p in prefs) else 0
        score = (1000 - (row.gi or 50)) + pref_boost*10 + max(0, 10 - abs(((lo+hi)/2) - row.carb_g))
        out.append((score, code))
    out.sort(key=lambda x: x[0], reverse=True)
    return [code for _, code in out[:k]]

# ---------------------------------------- 
# Meal-level nutrition helpers (ADD)
# ----------------------------------------
def _nutrition_from_recipe(recipe: dict) -> dict:
    # 레시피에 nutrition 요약이 있으면 사용, 없으면 0(보수적)
    n = recipe.get("nutrition") or {}
    def f(x): 
        try: return float(x)
        except: return 0.0
    return {
        "kcal": f(n.get("kcal", 0)),
        "carb_g": f(n.get("carb_g", 0)),
        "protein_g": f(n.get("protein_g", 0)),
        "fat_g": f(n.get("fat_g", 0)),
        "sugar_g": f(n.get("sugar_g", 0)),
        "sodium_mg": f(n.get("sodium_mg", 0)),
        "fiber_g": f(n.get("fiber_g", 0)),
        "gi": n.get("gi", None),
        "carb_weight": f(n.get("carb_g", 0)),
    }

def _nutrition_from_foodrow(row: FoodRow, serving_g: float) -> dict:
    ratio = max(0.0, serving_g) / 100.0  # FoodRow는 보통 100g 기준
    return {
        "kcal": row.kcal * ratio,
        "carb_g": row.carb_g * ratio,
        "protein_g": row.protein_g * ratio,
        "fat_g": row.fat_g * ratio,
        "sugar_g": row.sugar_g * ratio,
        "sodium_mg": row.sodium_mg * ratio,
        "fiber_g": row.fiber_g * ratio,
        "gi": row.gi,
        "carb_weight": row.carb_g * ratio,
    }

def _slot_default_serving(slot: str) -> float:
    # 슬롯별 기본 섭취량 가정(보수적)
    if slot == "breakfast": return 140.0
    if slot == "lunch": return 170.0
    if slot == "dinner": return 170.0
    if slot == "dessert": return 80.0
    return 150.0

def _sum_meal_nutrition(items: List[dict]) -> dict:
    total = {k: 0.0 for k in ["kcal","carb_g","protein_g","fat_g","sugar_g","sodium_mg","fiber_g","carb_weight"]}
    gis = []
    for n in items:
        for k in total.keys():
            v = n.get(k, 0.0)
            total[k] += 0.0 if v is None else float(v)
        if n.get("gi") is not None:
            gis.append((float(n["gi"]), float(n.get("carb_weight", 0.0))))
    if gis:
        wsum = sum(w for _, w in gis)
        if wsum > 0:
            total["gi_weighted"] = sum(gi*w for gi, w in gis) / wsum
        else:
            total["gi_weighted"] = sum(gi for gi, _ in gis)/len(gis)
    else:
        total["gi_weighted"] = None
    return total

def _rule_check_meal(total: dict, constraints: dict, meal_slot: str) -> Tuple[bool, Dict[str, Any]]:
    lo, hi = constraints.get("meal_carb_g", (35, 55))
    if meal_slot == "breakfast" and "breakfast_carb_g" in constraints:
        lo, hi = constraints["breakfast_carb_g"]

    ok = True
    report = {}

    def flag(key, cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            report[key] = msg

    flag("carb_g", lo <= total["carb_g"] <= hi, f"탄수 {total['carb_g']:.1f}g (목표 {lo}~{hi})")
    flag("kcal", total["kcal"] <= constraints.get("meal_kcal_max", 650), f"kcal {total['kcal']:.0f} > max")
    flag("sugar_g", total["sugar_g"] <= constraints.get("max_added_sugar_g", 10), f"당류 {total['sugar_g']:.1f} > max")
    flag("sodium_mg", total["sodium_mg"] <= constraints.get("sodium_mg_max", 1200), f"나트륨 {total['sodium_mg']:.0f} > max")

    if constraints.get("prefer_low_gi", True) and total.get("gi_weighted") is not None:
        flag("gi", total["gi_weighted"] <= 55, f"가중 GI {total['gi_weighted']:.0f} > 55")

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
            out.append({"type":"food","id":code,"name": row.name, "nutrition": n})
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

    def generate_weekly_plan(self, user_type: str, preferences: List[str], allergies: List[str], use_llm: bool = True) -> Dict[int, List[List[str]]]:
        constraints = self.knowledge.fetch_constraints(user_type)

        plan: Dict[int, List[List[str]]] = {}
        for day in range(7):
            day_meals: List[List[str]] = []
            for slot in MEAL_SLOTS:
                # 후보 넉넉히
                r_candidates = select_recipes(self.recipes, constraints, allergies, preferences, k=20, meal_slot=slot)
                f_candidates = select_foodcodes(self.food_index, constraints, allergies, preferences, k=20, meal_slot=slot)

                r_ranked = self._ranked_with_penalty(r_candidates, self.rec_mem, is_recipe=True)
                f_ranked = self._ranked_with_penalty(f_candidates, self.food_mem, is_recipe=False)

                # 동적 조합 선택
                r_need, f_need = self._choose_combo(slot)

                tries = 0
                chosen_ids: Optional[List[str]] = None
                MAX_TRIES_PER_MEAL = 8
                while tries < MAX_TRIES_PER_MEAL:
                    tries += 1
                    # 상위 N 내에서 소프트맥스 샘플링
                    r_pick = self._sample_bundle(r_ranked, self.rec_mem, count=r_need, topn=6, temperature=0.7) if r_need > 0 else []
                    f_pick = self._sample_bundle(f_ranked, self.food_mem, count=f_need, topn=6, temperature=0.7) if f_need > 0 else []

                    meal_items = self._materialize_items(r_pick, f_pick, slot)
                    ok, info = self._meal_ok(meal_items, constraints, slot, user_type, use_llm=use_llm)
                    if ok and meal_items:
                        chosen_ids = [*(x["id"] for x in meal_items)]
                        break

                    # 실패 시: 조합/온도/탑N을 가볍게 흔들어준다(탐색 폭 확대)
                    if tries in (3, 5):
                        # 조합 재선택으로 다양성 확보
                        r_need, f_need = self._choose_combo(slot)
                    if tries in (2, 4, 6):
                        # 탐욕성 완화
                        r_ranked = [(s*0.999, i) for (s,i) in r_ranked]
                        f_ranked = [(s*0.999, i) for (s,i) in f_ranked]

                # 폴백
                if not chosen_ids:
                    chosen_ids = (r_candidates[:1] or f_candidates[:1] or [])

                day_meals.append(chosen_ids)
            plan[day] = day_meals
        return plan
