# -*- coding: utf-8 -*-
"""
Low-Sugar Recipe Transformer (single file)
- 구조 파싱(ingredient/sauce/steps)
- 경계 인지 치환(안전): '청고추' → (보호), '매실청' → (대체)
- 온톨로지 태깅 → 카테고리별 치환 후보
- 분량 보정 메모/이유(explanations)
- 영양 합산 + GI/GL 추정
- 알러지 필터
- (선택) 지시문 안전 리라이팅

※ 외부 데이터:
- 음식 영양성분.xlsx, 원재료 영양성분.xlsx (경로는 환경변수/상수 참고)
"""
from __future__ import annotations
import os, re, json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

# -------------------------
# 경로 설정
# -------------------------
DATA_ROOT = os.getenv("APP_DATA_ROOT", "/app/data")
RECIPES_DIR = os.path.join(DATA_ROOT, "recipes/recipes")
NUTR_FOOD_PATH = os.path.join(DATA_ROOT, "share/음식 영양성분.xlsx")
NUTR_RAW_PATH  = os.path.join(DATA_ROOT, "share/원재료 영양성분.xlsx")

# -------------------------
# 공통 유틸
# -------------------------
def _deepcopy_json(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False))

def _norm(s: str) -> str:
    """간단 정규화(공백 제거·소문자 유지·특수기호 일부 정리)"""
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    return s

# -------------------------
# 1) 레시피 구조 도우미
# -------------------------
def strip_categories(recipe: Dict[str, Any]) -> Dict[str, Any]:
    out = _deepcopy_json(recipe)
    for k in ("category_time", "category_menu"):
        out.pop(k, None)
    return out

def ensure_recipe_schema(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """ingredient/sauce/steps 스키마 보정"""
    r = _deepcopy_json(recipe)
    r.setdefault("ingredient", {})
    r.setdefault("sauce", {})
    # steps는 문자열 리스트 또는 생략 가능
    steps = r.get("steps")
    if steps is None:
        r["steps"] = []
    elif isinstance(steps, str):
        r["steps"] = [steps]
    elif isinstance(steps, list):
        r["steps"] = [str(x) for x in steps]
    else:
        r["steps"] = [str(steps)]
    return r

# -------------------------
# 2) 온톨로지/태거
# -------------------------
# 보호 엔티티(치환 금지): 향채/고추류/기본 향신/육·난류 등
PROTECT_ENTITIES = {
    "청고추","청양고추","풋고추","홍고추","건고추","건홍고추","베트남고추","페퍼론치노","페퍼론치니",
    "고추","마늘","대파","쪽파","양파","부추","생강","후추","월계수잎","통후추","고수","바질","파슬리",
    "계란","달걀","닭가슴살","닭다리","소고기","돼지고기","오리고기","연어","새우","오징어","두부","계란흰자","계란노른자"
}
# '고추장'은 보호 X(복합 소스), 아래 카테고리에서 다룸

# 카테고리 사전(간단 키워드 기반)
CATS = {
    "sweetener": {"설탕","백설탕","흑설탕","갈색설탕","원당","슈가"},
    "syrup": {"물엿","올리고당","조청","시럽",
              "매실청","레몬청","유자청","생강청","자두청","배청","딸기청","자몽청","오미자청","허니시럽"},
    "high_gi_grain": {"흰쌀","백미","쌀밥","흰밥","맵쌀","찹쌀","쌀가루","멥쌀가루","떡","가래떡","인절미"},
    "flour": {"밀가루","박력분","중력분","강력분"},
    "breadcrumb": {"빵가루"},
    "starch": {"감자전분","옥수수전분","전분"},
    "noodle": {"라면","중면","소면","스파게티","파스타","국수","우동","칼국수","라멘","냉면"},
    "sugary_sauce": {"케첩","스위트칠리","데리야키","데리야끼","양념치킨","바베큐소스","허니머스타드","고추장"},
    "dairy": {"우유","연유","크림"},
    "yogurt": {"요거트","요구르트","플레인요거트","그릭요거트"},
}

# 별칭(정규화)
ALIASES = {
    "백설탕":"설탕","흑설탕":"설탕","갈색설탕":"설탕","설탕가루":"설탕",
    "케찹":"케첩","테리야끼":"데리야키","테리야키소스":"데리야키",
    "플레인요거트":"요거트","플레인요구르트":"요거트"
}

def canon_name(name: str) -> str:
    k = _norm(name)
    return ALIASES.get(k, name)

def classify(name: str) -> Optional[str]:
    """어절 경계가 애매한 한글을 대비해 '완전일치' 또는 '끝단/시작단' 휴리스틱"""
    base = canon_name(name)
    base_nospace = re.sub(r"\s+", "", base)
    # 보호 엔티티 우선
    if any(p == base or p == base_nospace for p in PROTECT_ENTITIES):
        return "protect"

    # 카테고리 검사(완전일치/부분일치)
    for cat, vocab in CATS.items():
        for kw in vocab:
            if base_nospace == kw or base.endswith(kw) or base.startswith(kw):
                return cat
    return None

# -------------------------
# 3) 카테고리별 치환 후보 & 분량/메모 로직
# -------------------------
@dataclass
class Swap:
    new_name: str
    qty_note: str  # 분량 보정 or 후처리 메모(문자열, UI에서 그대로 표시)
    reason: str    # 교체 이유 설명(로그/설명용)

def gen_swaps(name: str, qty: str) -> List[Swap]:
    """카테고리별 추천 치환안(간단 룰)"""
    cat = classify(name)
    if cat in (None, "protect"):
        return []

    swaps: List[Swap] = []
    if cat == "sweetener":
        swaps += [
            Swap("에리스리톨", "동량", "당류 대체(저흡수 당알코올)"),
            Swap("알룰로스", "1/2~2/3량", "갈변·점도 유지, 당류↓"),
            Swap("에리스리톨+스테비아", "에리스리톨 동량 + 스테비아 소량", "감미 보정 블렌드"),
        ]
    elif cat == "syrup":
        # ⚠️ '청고추' 등 오인 방지: 여기로 오면 이미 '매실청/유자청' 등 특정 청
        swaps += [
            Swap("알룰로스", "1/2~2/3량", "시럽류 당류↓"),
            Swap("무가당 과일퓨레", "동량", "자연 단맛·점도 보완"),
        ]
    elif cat == "high_gi_grain":
        swaps += [
            Swap("현미", "동량", "저GI 곡물 교체"),
            Swap("잡곡", "동량", "저GI 혼합곡"),
            Swap("콜리플라워 라이스", "동량", "탄수량↓ 대체"),
        ]
    elif cat == "flour":
        swaps += [
            Swap("통밀가루", "동량", "저GI 가루"),
            Swap("귀리 가루", "동량", "저GI 가루"),
            Swap("아몬드가루", "동량", "탄수량↓ 글루텐 無"),
        ]
    elif cat == "breadcrumb":
        swaps += [Swap("통밀빵가루", "동량", "저GI 빵가루")]
    elif cat == "starch":
        swaps += [Swap("타피오카전분", "소량", "점도 보정(과량 주의)"),
                  Swap("도토리전분", "소량", "점도 보정")]
    elif cat == "noodle":
        swaps += [
            Swap("통밀면", "동량(또는 20% 감량)", "저GI 면류 교체"),
            Swap("콩면", "동량", "탄수량↓ 단백질↑"),
            Swap("곤약면", "동량", "탄수량 크게↓"),
        ]
    elif cat == "sugary_sauce":
        if "고추장" in name:
            swaps += [
                Swap("저당 고추장", "동량", "가당 소스 저당 제품"),
                Swap("고추가루+간장+알룰로스+식초", "합성(레시피 메모 참조)", "무가당 조합 대체"),
            ]
        else:
            swaps += [
                Swap("무가당 토마토소스+식초", "동량", "시판 단맛 소스 대체"),
                Swap("저당 바베큐소스", "동량", "가당 소스 저당 제품"),
            ]
    elif cat == "dairy":
        swaps += [
            Swap("저지방 우유", "동량", "지방·당 저감"),
            Swap("무가당 두유", "동량", "유당↓ 대체"),
        ]
    elif cat == "yogurt":
        swaps += [Swap("무가당 그릭요거트", "동량", "당류↓ 단백질↑")]
    return swaps

# -------------------------
# 4) 안전 교체(경계 인지)
# -------------------------
def safe_replace_korean(text: str, old: str, new: str) -> str:
    """
    '설탕' -> '에리스리톨' 치환 시 조사 보존:
    ex) '설탕을' -> '에리스리톨을'
    """
    if not text:
        return text
    # 한글/영숫자 경계 기반 + 조사를 그룹으로 캡쳐
    patt = re.compile(
        r"(?<![가-힣A-Za-z0-9])" + re.escape(old) + r"(?P<josa>을|를|이|가|은|는|과|와|으로|로|에|에서|께|에게|랑)?(?![가-힣A-Za-z0-9])"
    )
    def _repl(m):
        j = m.group("josa") or ""
        return new + j
    return patt.sub(_repl, text)

def rewrite_steps(steps: List[str], mapping: Dict[str, str]) -> List[str]:
    out = []
    for line in steps:
        for old, new in mapping.items():
            if old == new:
                continue
            line = safe_replace_korean(line, old, new)
        out.append(line)
    return out

# -------------------------
# 5) 영양/GI 계산기
# -------------------------
@dataclass
class NutrRow:
    name: str
    per: float  # 기준량(g)
    carb: float
    prot: float
    fat: float
    sugar: float
    iron: float

class NutritionResolver:
    def __init__(self):
        self._df = pd.concat([self._load(NUTR_FOOD_PATH),
                              self._load(NUTR_RAW_PATH)], ignore_index=True, sort=False)
        self.name_col = self._pick(["식품명","원재료명","name"])
        self.base_col = self._pick(["기준량(g)","섭취단위(g)","base_g"])
        self.cols = {
            "carb": self._pick(["탄수화물(g)","carbohydrate_g","carb_g"]),
            "prot": self._pick(["단백질(g)","protein_g","prot_g"]),
            "fat":  self._pick(["지방(g)","fat_g"]),
            "sugar":self._pick(["당류(g)","sugars_g","sugar_g"]),
            "iron": self._pick(["철분(mg)","iron_mg"]),
        }
        self._cache: Dict[str, Optional[NutrRow]] = {}

    @staticmethod
    def _load(path: str) -> pd.DataFrame:
        try:
            if path.lower().endswith(".xlsx"):
                return pd.read_excel(path)
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def _pick(self, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in self._df.columns:
                return c
        return None

    @staticmethod
    def parse_qty_to_g(s: str) -> float:
        if not s:
            return float("nan")
        s = str(s)
        # 숫자 + g/ml
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(g|그램|ml|mL)", s)
        if m:
            return float(m.group(1))
        # 컵
        if "컵" in s:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*컵", s)
            n = float(m.group(1)) if m else 1.0
            return 200.0 * n
        # 숟가락/스푼
        if ("숟가락" in s) or ("스푼" in s):
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
            n = float(m.group(1)) if m else 1.0
            # '작은'이 포함되면 5g, 아니면 15g
            if "작" in s:
                return 5.0 * n
            return 15.0 * n
        # 숫자만
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
        if m:
            return float(m.group(1))
        return float("nan")

    def lookup(self, name: str) -> Optional[NutrRow]:
        key = _norm(name)
        if key in self._cache:
            return self._cache[key]
        if self._df.empty or not self.name_col:
            self._cache[key] = None
            return None
        # 부분/공백 제거 일치
        ser = self._df[self.name_col].astype(str)
        cand = self._df[ser.str.contains(name, regex=False, na=False)]
        if cand.empty:
            no_space = name.replace(" ", "")
            cand = self._df[ser.str.replace(" ","", regex=False).str.contains(no_space, regex=False, na=False)]
        if cand.empty:
            self._cache[key] = None
            return None
        row = cand.iloc[0]
        base = float(row.get(self.base_col, 100) or 100)
        out = NutrRow(
            name=str(row[self.name_col]),
            per=base,
            carb=float(row.get(self.cols["carb"], 0) or 0),
            prot=float(row.get(self.cols["prot"], 0) or 0),
            fat =float(row.get(self.cols["fat"], 0) or 0),
            sugar=float(row.get(self.cols["sugar"], 0) or 0),
            iron=float(row.get(self.cols["iron"], 0) or 0),
        )
        self._cache[key] = out
        return out

    def totalize(self, ingredient: Dict[str, str]) -> Dict[str, float]:
        total = dict(carb=0.0, prot=0.0, fat=0.0, sugar=0.0, iron=0.0)
        for name, qty in (ingredient or {}).items():
            qty_g = self.parse_qty_to_g(str(qty))
            row = self.lookup(name)
            if not row:
                continue
            mult = (qty_g / row.per) if (qty_g == qty_g and row.per) else 1.0  # NaN 체크
            total["carb"]  += row.carb  * mult
            total["prot"]  += row.prot  * mult
            total["fat"]   += row.fat   * mult
            total["sugar"] += row.sugar * mult
            total["iron"]  += row.iron  * mult
        return total

class GIResolver:
    """간단 GI/GL 추정"""
    def __init__(self):
        self.map = {
            "현미":55, "잡곡":50, "퀴노아":53, "귀리":55, "통밀":58,
            "흰쌀":73, "백미":73, "찹쌀":82, "감자":78, "고구마":63,
            "면":70, "콩면":35, "두부":15, "우유":47, "요거트":36,
            "사과":38, "바나나":52, "당근":47, "브로콜리":15,
            "빵":75, "통밀빵":62, "현미밥":55, "콜리플라워":10,
            "올리고당":90, "설탕":65, "꿀":61, "에리스리톨":0, "스테비아":0, "알룰로스":0,
            "고추장":66  # 대략, 시판 가당 소스 가정
        }

    def gi_for_name(self, name: str) -> Optional[float]:
        nm = name.replace(" ", "")
        # 구체→일반 키 순으로 탐색
        for k in sorted(self.map.keys(), key=len, reverse=True):
            if k in nm:
                return float(self.map[k])
        return None

    def compute_gi_gl(self, ingredient: Dict[str, str], nutr: NutritionResolver) -> Tuple[float, float]:
        """
        GI(weighted) & GL(total) 계산:
        - 각 재료 탄수화물량 × GI/100 을 합산(GL)
        - GI는 탄수 비중 가중평균
        """
        total_carb = 0.0
        weighted_gi = 0.0
        total_gl = 0.0
        for name, qty in (ingredient or {}).items():
            row = nutr.lookup(name)
            if not row:
                continue
            qty_g = nutr.parse_qty_to_g(str(qty))
            mult = (qty_g / row.per) if (qty_g == qty_g and row.per) else 1.0
            carb = row.carb * mult
            gi = self.gi_for_name(name)
            if gi is None:
                continue
            total_carb += carb
            weighted_gi += gi * carb
            total_gl += carb * gi / 100.0
        if total_carb > 0:
            gi_val = max(10.0, min(95.0, weighted_gi / total_carb))
        else:
            gi_val = 45.0
        return round(gi_val, 1), round(total_gl, 1)

# -------------------------
# 6) 알러지 필터
# -------------------------
def apply_allergy_filter(recipe: Dict[str, Any], allergies: List[str]) -> Dict[str, Any]:
    if not allergies:
        return recipe
    r = _deepcopy_json(recipe)
    def _f(sec: Dict[str,str]) -> Dict[str,str]:
        out = {}
        for k, v in (sec or {}).items():
            if any(a and a in k for a in allergies):
                continue
            out[k] = v
        return out
    r["ingredient"] = _f(r.get("ingredient", {}))
    r["sauce"]      = _f(r.get("sauce", {}))
    return r

# -------------------------
# 7) 메인 변환(안전 치환)
# -------------------------
@dataclass
class ChangeLog:
    before: str
    after: str
    qty_note: str
    reason: str
    section: str  # ingredient/sauce

def apply_low_sugar_transform(recipe: Dict[str, Any], user_type: str) -> Tuple[Dict[str, Any], List[ChangeLog], Dict[str,str]]:
    """
    - ingredient/sauce에서만 치환
    - 보호 엔티티·향채 등은 스킵
    - 고유명 혼동 방지: 카테고리 태깅 기반 교체
    """
    r = _deepcopy_json(recipe)
    changes: List[ChangeLog] = []
    name_map_for_steps: Dict[str,str] = {}  # steps 리네이밍용

    def _transform_section(section: Dict[str, str], sec_name: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name, qty in (section or {}).items():
            # 중복 '(대체)' 방지
            disp_name = name.replace("(대체)","").strip()

            # 보호/카테고리 판정
            tag = classify(disp_name)
            if tag in (None, "protect"):
                out[disp_name] = qty
                continue

            # 후보 생성
            cands = gen_swaps(disp_name, qty)
            if not cands:
                out[disp_name] = qty
                continue

            # 간단 선택 휴리스틱: 알룰로스 우선(시럽/감미), 그 외 첫 후보
            pick: Swap = None  # type: ignore
            if tag in {"sweetener","syrup"}:
                # 사용자 타입 보정(식후형은 강하게)
                if user_type.upper() == "PPG_HIGH":
                    for sw in cands:
                        if "알룰로스" in sw.new_name:
                            pick = sw; break
                if pick is None:
                    for sw in cands:
                        if "에리스리톨" in sw.new_name:
                            pick = sw; break
            if pick is None:
                pick = cands[0]

            new_name = pick.new_name
            # 결과 기록
            out["(대체) " + new_name] = str(qty) + (f" [{pick.qty_note}]" if pick.qty_note else "")
            changes.append(ChangeLog(before=disp_name, after=new_name, qty_note=pick.qty_note, reason=pick.reason, section=sec_name))
            name_map_for_steps[disp_name] = new_name
        return out

    r["ingredient"] = _transform_section(r.get("ingredient", {}), "ingredient")
    r["sauce"]      = _transform_section(r.get("sauce", {}), "sauce")

    # user_type 메모
    r.setdefault("notes", [])
    if user_type.upper() == "PPG_HIGH":
        r["notes"].append("식후혈당형: 전분/당 대체 강도↑, 식초 1작은술 권장")
        if "식초 소량" not in r.get("sauce", {}):
            r.setdefault("sauce", {})["식초 소량"] = "1작은술"
    elif user_type.upper() == "FPG_HIGH":
        r["notes"].append("공복혈당형: 저GI 곡물 유지 및 야식 금지 권장")
    elif user_type.upper() == "WEIGHT_GAIN":
        r["notes"].append("체중증가형: 유지/기름 20% 감량 권장")
        for k in list(r.get("sauce", {}).keys()):
            if any(x in k for x in ["오일","기름","버터","마요"]):
                r["sauce"][k] = f"{r['sauce'][k]} (20% 감량)"
    elif user_type.upper() == "INSULIN":
        r["notes"].append("인슐린형: 저GI 유지 + 단당 제한")

    return r, changes, name_map_for_steps

# -------------------------
# 8) 외부 인터페이스
# -------------------------
def transform_recipe_from_dict(food_json: Dict[str, Any], user_type: str, allergies: List[str]) -> Dict[str, Any]:
    """
    기존 시그니처와 호환되는 안전 변환 버전
    """
    # 1) 정리
    recipe0 = ensure_recipe_schema(strip_categories(food_json))

    # 2) 안전 저당 치환
    recipe1, changes, name_map = apply_low_sugar_transform(recipe0, user_type=user_type)

    # 3) 알러지 필터
    recipe2 = apply_allergy_filter(recipe1, allergies)

    # 4) 지시문 리라이팅(선택)
    if recipe2.get("steps"):
        recipe2["steps"] = rewrite_steps(recipe2["steps"], name_map)

    # 5) 영양 합산
    nutr = NutritionResolver()
    totals = nutr.totalize(recipe2.get("ingredient", {}))

    # 6) GI/GL
    gi_res = GIResolver()
    GI_VAL, GL_VAL = gi_res.compute_gi_gl(recipe2.get("ingredient", {}), nutr)

    # 7) 설명/변경사항 요약
    explanations = [
        f"[{c.section}] {c.before} → {c.after} (분량: {c.qty_note}) · 이유: {c.reason}"
        for c in changes
    ]

    # 8) 출력 포맷(기존 키 + 확장)
    out = {
        "food": recipe2,
        "GI_VAL": GI_VAL,                 # 추정 GI (가중평균)
        "GL_VAL": GL_VAL,                 # 총 GL (합산)
        "CH_VAL": round(totals.get("carb", 0.0), 1),
        "PR_VAL": round(totals.get("prot", 0.0), 1),
        "FAT_VAL": round(totals.get("fat", 0.0), 1),
        "IC_VAL": round(totals.get("iron", 0.0), 2),
        "explanations": explanations,     # 변경 근거(문장)
        "audit": {
            "rules_passed": True,
            "warnings": []  # 후속: kcal 편차/나트륨 등 검증 경고 추가 가능
        }
    }
    return out

