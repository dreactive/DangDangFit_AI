# file: validate_weekly_plan.py
from __future__ import annotations
import json, math, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# --- 프로젝트 모듈: diet_service.py 를 그대로 활용 ---
#    (동일 경로/동일 함수명 기준)
from diet_service import (
    DataPaths, DietPlanner, MEAL_SLOTS,
    _nutrition_from_recipe, _nutrition_from_foodrow,
    _slot_default_serving, _sum_meal_nutrition, _rule_check_meal
)

# -----------------------------
# 검증 유틸
# -----------------------------
def _finite(x) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isfinite(x)
    except Exception:
        return False

def _gi_gl_for_total(total: dict) -> Tuple[Optional[float], Optional[float]]:
    """가중 GI와 슬롯 GL 계산 (GL = GI_weighted * carb_g / 100)"""
    giw = total.get("gi_weighted", None)
    carb = total.get("carb_g", None)
    if _finite(giw) and _finite(carb):
        return float(giw), float(giw) * float(carb) / 100.0
    return (giw if _finite(giw) else None, None)

def _materialize_from_ids(planner: DietPlanner, ids: List[str], slot: str) -> List[dict]:
    """추천 결과의 ids를 실제 아이템(레시피/식품) + 영양으로 복원"""
    rec_map = {r["_file"]: r for r in planner.recipes}
    items: List[dict] = []
    default_sv = _slot_default_serving(slot)
    for _id in ids:
        if _id in rec_map:
            r = rec_map[_id]
            n = _nutrition_from_recipe(r)
            items.append({"type": "recipe", "id": _id, "name": r.get("title") or r.get("name") or _id, "nutrition": n})
        else:
            row = planner.food_index.get(_id)
            if not row:
                # 알 수 없는 ID는 스킵(로그만 남김)
                print(f"[WARN] Unknown ID in plan: {slot} -> {_id}")
                continue
            n = _nutrition_from_foodrow(row, serving_g=default_sv)
            items.append({"type": "food", "id": _id, "name": row.name, "serving_g": default_sv, "nutrition": n})
    return items

# 추가 부분

import math

def _round_if_finite(v, nd=2):
    try:
        f = float(v)
        if math.isfinite(f):
            return round(f, nd)
    except (TypeError, ValueError):
        pass
    return None  # None/NaN은 JSON에서 null로 남김



@dataclass
class SlotResult:
    slot: str
    ok: bool
    report: Dict[str, Any]
    total: Dict[str, float]
    gi_weighted: Optional[float]
    gl: Optional[float]
    ids: List[str]
    names: List[str]

@dataclass
class DayResult:
    day_idx: int
    slots: List[SlotResult]
    totals: Dict[str, float]           # 하루 합계
    gi_weighted_list: List[float]      # 슬롯별 GIw(결측 제외)
    gl_sum: Optional[float]            # 하루 GL 합
    macro_ratio: Dict[str, float]      # 탄/단/지 비율(%)

@dataclass
class WeekSummary:
    constraint_pass_rate: float
    carb_mae: float
    high_gl_ratio: float
    giw_median: Optional[float]
    limit_violation_rate: float        # kcal/sodium/sugar 상한 초과 슬롯 비율
    diversity_unique_ids: int
    total_slots: int

# -----------------------------
# 검증 본체
# -----------------------------
class MealPlanValidator:
    def __init__(self, paths: Optional[DataPaths] = None):
        self.paths = paths or DataPaths()
        self.planner = DietPlanner(self.paths)
        # 제약(케이스 유형은 평가 대상과 동일하게 맞춰야 함)
        self._constraints_cache: Dict[str, Dict[str, Any]] = {}

    def _constraints(self, user_type: str) -> Dict[str, Any]:
        if user_type not in self._constraints_cache:
            self._constraints_cache[user_type] = self.planner.knowledge.fetch_constraints(user_type)
        return self._constraints_cache[user_type]

    def _rebalance_like_model(self, items: List[dict], slot: str, user_type: str) -> List[dict]:
        # 모델과 동일한 탄수 목표 보정 사용(식품 serving만 조정)
        constraints = self._constraints(user_type)
        try:
            items = self.planner._rebalance_servings(items, constraints, slot)  # 내부 메서드 재사용
        except Exception:
            pass
        return items

    def _eval_slot(self, ids: List[str], slot: str, user_type: str, do_rebalance=True) -> SlotResult:
        constraints = self._constraints(user_type)
        items = _materialize_from_ids(self.planner, ids, slot)
        if do_rebalance:
            items = self._rebalance_like_model(items, slot, user_type)

        total = _sum_meal_nutrition([it["nutrition"] for it in items])
        ok, report = _rule_check_meal(total, constraints, slot)
        giw, gl = _gi_gl_for_total(total)

        return SlotResult(
            slot=slot,
            ok=ok,
            report=report,
            total=total,
            gi_weighted=giw,
            gl=gl,
            ids=ids,
            names=[it["name"] for it in items]
        )

    def _eval_day(self, day_idx: int, day_meals: List[List[str]], user_type: str) -> DayResult:
        # day_meals: [breakfast_ids, lunch_ids, dinner_ids, dessert_ids]
        slot_results: List[SlotResult] = []
        for slot, ids in zip(MEAL_SLOTS, day_meals):
            slot_results.append(self._eval_slot(ids, slot, user_type, do_rebalance=True))

        # 하루 합계
        day_total = {k: 0.0 for k in ["kcal","carb_g","protein_g","fat_g","sugar_g","sodium_mg","fiber_g","carb_weight"]}
        gi_list, gl_list = [], []
        for sr in slot_results:
            for k in day_total.keys():
                day_total[k] += float(sr.total.get(k, 0.0) or 0.0)
            if _finite(sr.gi_weighted):
                gi_list.append(float(sr.gi_weighted))
            if _finite(sr.gl):
                gl_list.append(float(sr.gl))

        gl_sum = sum(gl_list) if gl_list else None

        kcal = max(1e-6, day_total["kcal"])
        carb_pct = (day_total["carb_g"] * 4 / kcal) * 100.0
        protein_pct = (day_total["protein_g"] * 4 / kcal) * 100.0
        fat_pct = (day_total["fat_g"] * 9 / kcal) * 100.0

        return DayResult(
            day_idx=day_idx,
            slots=slot_results,
            totals=day_total,
            gi_weighted_list=gi_list,
            gl_sum=gl_sum,
            macro_ratio={"carb_pct": carb_pct, "protein_pct": protein_pct, "fat_pct": fat_pct}
        )

    def _week_summary(self, days: List[DayResult], user_type: str) -> WeekSummary:
        constraints = self._constraints(user_type)

        # 1) 제약 충족률
        all_slots = [sr for d in days for sr in d.slots]
        total_slots = len(all_slots)
        pass_cnt = sum(1 for sr in all_slots if sr.ok)
        pass_rate = (pass_cnt / total_slots * 100.0) if total_slots else 0.0

        # 2) 탄수 목표 오차(MAE): 슬롯별 |carb - (lo+hi)/2|
        def _slot_mid(slot: str):
            lo, hi = constraints.get(f"{slot}_carb_g", constraints.get("meal_carb_g", (35,55)))
            return (lo + hi) / 2.0
        abs_err = []
        for sr in all_slots:
            mid = _slot_mid(sr.slot)
            if _finite(sr.total.get("carb_g")):
                abs_err.append(abs(sr.total["carb_g"] - mid))
        carb_mae = (sum(abs_err)/len(abs_err)) if abs_err else 0.0

        # 3) High GL 비중(슬롯 GL >= 20)
        gl_flags = [1 for sr in all_slots if _finite(sr.gl) and sr.gl >= 20.0]
        gl_known = [1 for sr in all_slots if _finite(sr.gl)]
        high_gl_ratio = (sum(gl_flags)/len(gl_known)*100.0) if gl_known else 0.0

        # 4) GIw 중앙값
        gi_vals = [sr.gi_weighted for sr in all_slots if _finite(sr.gi_weighted)]
        gi_vals.sort()
        giw_median = None
        if gi_vals:
            n = len(gi_vals)
            giw_median = gi_vals[n//2] if n % 2 == 1 else (gi_vals[n//2 - 1] + gi_vals[n//2]) / 2.0

        # 5) 상한 초과율(kcal/sodium/sugar)
        limit_viol = 0
        for sr in all_slots:
            rpt = sr.report or {}
            if "kcal" in rpt or "sodium_mg" in rpt or "sugar_g" in rpt:
                limit_viol += 1
        limit_violation_rate = (limit_viol / total_slots * 100.0) if total_slots else 0.0

        # 6) 다양성: 고유 id 수
        uniq_ids = set()
        for sr in all_slots:
            for _id in sr.ids:
                uniq_ids.add(_id)

        return WeekSummary(
            constraint_pass_rate=pass_rate,
            carb_mae=carb_mae,
            high_gl_ratio=high_gl_ratio,
            giw_median=giw_median,
            limit_violation_rate=limit_violation_rate,
            diversity_unique_ids=len(uniq_ids),
            total_slots=total_slots
        )

    # -----------------------------
    # 공개 API
    # -----------------------------
    def validate_generated(self, user_type: str, preferences: List[str], allergies: List[str]) -> Dict[str, Any]:
        """모델로 주간 식단을 생성하고 곧장 검증"""
        plan = self.planner.generate_weekly_plan(user_type=user_type, preferences=preferences, allergies=allergies, use_llm=False)
        return self.validate_plan(plan, user_type)

    def validate_plan(self, plan: Dict[int, List[List[str]]], user_type: str) -> Dict[str, Any]:
        """이미 생성된 주간 식단(plan)을 검증"""
        days: List[DayResult] = []
        for d in range(7):
            day_meals = plan.get(d) or [[],[],[],[]]
            days.append(self._eval_day(d, day_meals, user_type))

        summary = self._week_summary(days, user_type)

        # 직렬화 가능한 결과로 변환
        out = {
            "user_type": user_type,
            "summary": {
                "constraint_pass_rate_percent": round(summary.constraint_pass_rate, 2),
                "carb_mae_g": round(summary.carb_mae, 2),
                "high_gl_ratio_percent": round(summary.high_gl_ratio, 2),
                "gi_weighted_median": (round(summary.giw_median, 2) if summary.giw_median is not None else None),
                "limit_violation_rate_percent": round(summary.limit_violation_rate, 2),
                "diversity_unique_ids": summary.diversity_unique_ids,
                "total_slots": summary.total_slots,
            },
            "days": []
        }
        for dr in days:
            out["days"].append({
                "day": dr.day_idx,
                "totals": {k: _round_if_finite(v, 2) for k, v in dr.totals.items()}, # 수정됨.
                "macro_ratio_percent": {k: _round_if_finite(v, 1) for k, v in dr.macro_ratio.items()}, # 수정됨.
                "gl_sum": (round(float(dr.gl_sum), 2) if dr.gl_sum is not None else None),
                "slots": [
                    {
                        "slot": sr.slot,
                        "ok": sr.ok,
                        "ids": sr.ids,
                        "names": sr.names,
                        "total": {k: _round_if_finite(v, 2) for k, v in sr.total.items()}, # 수정됨.
                        "gi_weighted": (round(float(sr.gi_weighted), 2) if _finite(sr.gi_weighted) else None),
                        "gl": (round(float(sr.gl), 2) if _finite(sr.gl) else None),
                        "report": sr.report
                    } for sr in dr.slots
                ]
            })
        return out

# -----------------------------
# CLI/예시 실행
# -----------------------------
if __name__ == "__main__":
    """
    사용 예시 1) 모델이 생성한 식단을 즉시 검증:
      python validate_weekly_plan.py

    사용 예시 2) 저장된 주간 식단 JSON을 검증:
      python validate_weekly_plan.py --plan plan.json --user_type P
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_type", default="P", help="F/P/E/I 중 하나")
    parser.add_argument("--preferences", nargs="*", default=[], help="선호 키워드")
    parser.add_argument("--allergies", nargs="*", default=[], help="알레르겐 키워드")
    parser.add_argument("--plan", type=str, default=None, help="이미 생성된 주간 식단 JSON 경로")
    parser.add_argument("--out_json", type=str, default="validation_report.json")
    parser.add_argument("--out_csv", type=str, default=None, help="슬롯별 상세를 CSV로 저장할 경로(옵션)")
    args = parser.parse_args()

    validator = MealPlanValidator(paths=DataPaths())  # 경로 필요 시 DataPaths(...)로 교체

    if args.plan:
        with open(args.plan, "r", encoding="utf-8") as f:
            plan = json.load(f)
        report = validator.validate_plan(plan, user_type=args.user_type)
    else:
        report = validator.validate_generated(
            user_type=args.user_type,
            preferences=args.preferences,
            allergies=args.allergies
        )

    # JSON 저장
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved JSON report -> {args.out_json}")

    # CSV(옵션) 저장
    if args.out_csv:
        rows = []
        for d in report["days"]:
            for s in d["slots"]:
                rows.append({
                    "day": d["day"],
                    "slot": s["slot"],
                    "ok": s["ok"],
                    "names": " | ".join(s["names"]),
                    "carb_g": s["total"]["carb_g"],
                    "kcal": s["total"]["kcal"],
                    "sodium_mg": s["total"]["sodium_mg"],
                    "sugar_g": s["total"]["sugar_g"],
                    "gi_weighted": s["gi_weighted"] if s["gi_weighted"] is not None else "",
                    "gl": s["gl"] if s["gl"] is not None else "",
                    "report_keys": ",".join(s["report"].keys())
                })
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] Saved CSV detail -> {args.out_csv}")
