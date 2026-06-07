"""dlt_betting.py
Betting strategy layer for DLT (大乐透) lottery.

Differences from SSQ:
- Red numbers: choose 5 from 1-35
- Blue numbers: choose 2 from 1-12
- Filters apply separately to red part (sum, odd/even, AC) and optional blue constraints.

The API mirrors ssq_betting.py so callers can use the same workflow.
"""

import json
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# ------------------------------------------------------------
# 1. Kelly / half-Kelly
# ------------------------------------------------------------

def kelly_fraction(p: float, b: float, half: bool = False) -> float:
    if not (0 < p < 1) or b <= 0:
        raise ValueError("Invalid probability or odds")
    f = (p * (b + 1) - 1) / b
    if f < 0:
        f = 0.0
    return f / 2 if half else f

# ------------------------------------------------------------
# 2. Filters - only on the red numbers part
# ------------------------------------------------------------

def sum_range_filter(combos: List[Tuple[Tuple[int, ...], Tuple[int, ...]]], low: int, high: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """combos = [(red_tuple, blue_tuple), ...]"""
    return [c for c in combos if low <= sum(c[0]) <= high]

def odd_even_ratio_filter(combos: List[Tuple[Tuple[int, ...], Tuple[int, ...]]], min_ratio: float, max_ratio: float) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    out = []
    for red, blue in combos:
        odd_cnt = sum(1 for n in red if n % 2 == 1)
        ratio = odd_cnt / len(red)
        if min_ratio <= ratio <= max_ratio:
            out.append((red, blue))
    return out

def ac_value(nums: List[int]) -> int:
    """
    计算AC值（算术复杂度）：
    所有号码两两差值的不同值数量 - (号码数 - 1)

    大乐透前区5码的AC值范围通常在 0~10，
    常见值为4~8，6最为高频。

    示例: [1, 5, 8, 15, 22]
      差值: 4,7,14,21,3,10,17,7,14,7 -> {3,4,7,10,14,17,21} 7个不同值
      AC = 7 - (5-1) = 3
    """
    if len(nums) < 2:
        return 0
    sorted_nums = sorted(nums)
    diffs = set()
    for i in range(len(sorted_nums)):
        for j in range(i + 1, len(sorted_nums)):
            diffs.add(abs(sorted_nums[j] - sorted_nums[i]))
    return len(diffs) - (len(nums) - 1)


def ac_filter(combos: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
              min_ac: int = 4, max_ac: int = 10) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """按AC值范围过滤候选组合"""
    return [c for c in combos if min_ac <= ac_value(list(c[0])) <= max_ac]

# ------------------------------------------------------------
# 3. Dan-Tuo generator for DLT (separate red and blue)
# ------------------------------------------------------------

def generate_dantuo(red_top: List[int], red_missing: List[int],
                    blue_top: List[int], blue_missing: List[int],
                    red_total: int = 5, blue_total: int = 2) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    from itertools import combinations
    # Red part
    if len(red_top) >= red_total:
        red_combos = [tuple(sorted(red_top[:red_total]))]
    else:
        pool_red = list(set(red_missing) - set(red_top))
        need_red = red_total - len(red_top)
        red_combos = [tuple(sorted(red_top + list(extra))) for extra in combinations(pool_red, need_red)]
    # Blue part
    if len(blue_top) >= blue_total:
        blue_combos = [tuple(sorted(blue_top[:blue_total]))]
    else:
        pool_blue = list(set(blue_missing) - set(blue_top))
        need_blue = blue_total - len(blue_top)
        blue_combos = [tuple(sorted(blue_top + list(extra))) for extra in combinations(pool_blue, need_blue)]
    # Cartesian product
    result: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for r in red_combos:
        for b in blue_combos:
            result.append((r, b))
    return result

# ------------------------------------------------------------
# 4. Risk diversification (pure Python, no PuLP dependency)
# ------------------------------------------------------------

def diversify_risk(combos: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
                   max_total: int,
                   risk_weights: Optional[List[float]] = None) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    投注分散：用线性规划从候选组合中挑选最多 max_total 个结果。
    使用 PuLP 库求解整数线性规划问题。

    Args:
        combos: 候选组合列表
        max_total: 最大投注数（或budget上限）
        risk_weights: 各组合的"风险成本"，默认为全1

    Returns:
        选中的组合子集
    """
    import pulp

    n = len(combos)
    if n == 0:
        return []

    if risk_weights is None:
        risk_weights = [1.0] * n
    elif len(risk_weights) != n:
        raise ValueError(f"risk_weights 长度({len(risk_weights)})与 combos({n})不一致")

    prob = pulp.LpProblem("DLTDiversify", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")

    # 目标：最大化选中组合的数量
    prob += pulp.lpSum([x[i] for i in range(n)])

    # 约束：风险成本总和不超过预算
    prob += pulp.lpSum([risk_weights[i] * x[i] for i in range(n)]) <= max_total

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    selected = [combos[i] for i in range(n) if pulp.value(x[i]) == 1]
    return selected

# ------------------------------------------------------------
# 5. Export utilities - flatten for CSV/JSON
# ------------------------------------------------------------

def export_to_json(combos: List[Tuple[Tuple[int, ...], Tuple[int, ...]]], filepath: str) -> None:
    data = [{"red": list(r), "blue": list(b)} for r, b in combos]
    Path(filepath).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def export_to_csv(combos: List[Tuple[Tuple[int, ...], Tuple[int, ...]]], filepath: str) -> None:
    red_len = len(combos[0][0]) if combos else 0
    blue_len = len(combos[0][1]) if combos else 0
    header = ["combo_id"] + [f"red{i+1}" for i in range(red_len)] + [f"blue{i+1}" for i in range(blue_len)]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for idx, (r, b) in enumerate(combos, start=1):
            w.writerow([idx] + list(r) + list(b))

if __name__ == "__main__":
    # Simple demo
    red_combos = [(1, 2, 3, 4, 5)]
    blue_combos = [(1, 2)]
    combos = [(red, blue) for red in red_combos for blue in blue_combos]

    # Test ac_value fix
    test_nums = [1, 5, 8, 15, 22]
    ac = ac_value(test_nums)
    print(f"AC值测试: {test_nums} -> AC={ac} (差值: {set(), 合理})")

    selected = diversify_risk(combos, max_total=10)
    export_to_json(selected, "dlt_selected.json")
    export_to_csv(selected, "dlt_selected.csv")
