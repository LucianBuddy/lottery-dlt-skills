#!/usr/bin/env python3
"""
大乐透预测结果存储模块
自动保存预测结果至知识库，保留最近五期。

设计原则：
1. 预测完成后自动调用 store_prediction() 保存
2. 超过5期的数据自动清除
3. 对比时调用 load_prediction(period) 或 compare_with_actual() 读取
"""

# 保留期数
MAX_PERIODS = 5

import json
import os
from typing import Optional, Dict, Any, List

# 存储文件路径（基于技能包目录）
STORE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..',
    'memory', 'lottery_predictions.json'
)


def _ensure_store():
    """确保存储文件存在"""
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    if not os.path.exists(STORE_PATH):
        with open(STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump({"predictions": []}, f, ensure_ascii=False, indent=2)


def load_all() -> List[Dict[str, Any]]:
    """读取所有已存储的预测记录"""
    _ensure_store()
    try:
        with open(STORE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("predictions", [])
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_all(predictions: List[Dict[str, Any]]):
    """覆写全部预测记录"""
    _ensure_store()
    with open(STORE_PATH, 'w', encoding='utf-8') as f:
        json.dump({"predictions": predictions}, f, ensure_ascii=False, indent=2)


def store_prediction(
    period: str,
    front_bets: List[Dict[str, Any]],
    compound_bets: Optional[Dict[str, List[Dict[str, Any]]]] = None
):
    """
    存储一期预测结果，自动清理超过5期的旧数据。

    Args:
        period: 期号字符串，如 "26059"
        front_bets: 单注列表 [{'front': [...], 'back': [...], ...}, ...]
        compound_bets: 复式投注字典 {类型名称: [{'front': [...], 'back': [...]}, ...]}
    """
    predictions = load_all()

    # 删除已存在的同期限记录（防止重复）
    predictions = [p for p in predictions if p.get("period") != period]

    # 添加新记录
    entry = {
        "period": period,
        "single_bets": [
            {
                "front": bet.get("front", []),
                "back": bet.get("back", []),
                "final_score": round(bet.get("final_score", 0), 4),
            }
            for bet in front_bets
        ],
    }
    if compound_bets:
        compound_clean = {}
        for btype, bets in compound_bets.items():
            compound_clean[btype] = [
                {
                    "front": bet.get("front", []),
                    "back": bet.get("back", []),
                }
                for bet in bets
            ]
        entry["compound_bets"] = compound_clean

    predictions.append(entry)

    # 仅保留最近5期，按期号排序后取最后5条
    predictions.sort(key=lambda x: x.get("period", "0"))
    predictions = predictions[-5:]

    save_all(predictions)

    kept_periods = [p["period"] for p in predictions]
    print(f"[PredictionStore] ✅ 已保存 {period} 期预测 | "
          f"共 {len(predictions)} 条记录 (留存: {kept_periods})")


def load_prediction(period: str) -> Optional[Dict[str, Any]]:
    """
    从知识库读取指定期号的预测结果。

    Args:
        period: 期号字符串

    Returns:
        Dict 或 None（未找到）
    """
    predictions = load_all()
    for p in predictions:
        if p.get("period") == period:
            return p
    return None


def list_saved_periods() -> List[str]:
    """列出知识库中所有已保存的期号（按期号排序，最新在前）"""
    predictions = load_all()
    periods = [p.get("period", "?") for p in predictions]
    periods.sort(reverse=True)
    return periods


def compare_with_actual(
    period: str,
    actual_front: List[int],
    actual_back: List[int]
) -> Optional[Dict[str, Any]]:
    """
    对比某期预测与实际开奖号码。

    Args:
        period: 期号
        actual_front: 实际前区号码 [n1,n2,n3,n4,n5]
        actual_back: 实际后区号码 [n1,n2]

    Returns:
        Dict 含 {'period', 'predicted', 'actual', 'hits', 'best_hit', 'compound_hits'}
        或 None（未找到对应预测）
    """
    pred = load_prediction(period)
    if not pred:
        return None

    actual_f_set = set(actual_front)
    actual_b_set = set(actual_back)

    # 单式命中统计
    single_hits = []
    best_combined = 0
    best_single = None
    for bet in pred.get('single_bets', []):
        f_hit = len(set(bet.get('front', [])) & actual_f_set)
        b_hit = len(set(bet.get('back', [])) & actual_b_set)
        total = f_hit + b_hit
        single_hits.append({
            'front': bet['front'],
            'back': bet['back'],
            'front_hit': f_hit,
            'back_hit': b_hit,
            'total': total,
            'score': bet.get('final_score', 0),
        })
        if total > best_combined:
            best_combined = total
            best_single = {'front': bet['front'], 'back': bet['back'],
                           'front_hit': f_hit, 'back_hit': b_hit, 'total': total}

    # 复式命中统计
    compound_hits = {}
    for ctype, bets in pred.get('compound_bets', {}).items():
        for bet in bets:
            f_hit = len(set(bet.get('front', [])) & actual_f_set)
            b_hit = len(set(bet.get('back', [])) & actual_b_set)
            if ctype not in compound_hits or f_hit + b_hit > compound_hits[ctype]['total']:
                compound_hits[ctype] = {
                    'front_hit': f_hit,
                    'back_hit': b_hit,
                    'total': f_hit + b_hit,
                }

    return {
        'period': period,
        'predicted': {
            'single_count': len(pred.get('single_bets', [])),
            'has_compound': 'compound_bets' in pred,
        },
        'actual': {
            'front': sorted(actual_front),
            'back': sorted(actual_back),
        },
        'single_hits': single_hits,
        'best_single': best_single,
        'best_combined_hits': best_combined,
        'compound_best_hits': compound_hits,
    }


def show_store():
    """展示存储文件路径和内容摘要"""
    predictions = load_all()
    return {
        "path": STORE_PATH,
        "count": len(predictions),
        "periods": [p.get("period", "?") for p in predictions],
    }
