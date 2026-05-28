#!/usr/bin/env python3
"""
大乐透预测结果存储模块
自动保存预测结果至知识库，仅保留最近两期。

设计原则：
1. 预测完成后自动调用 store_prediction() 保存
2. 超过2期的数据自动清除
3. 对比时调用 load_prediction(period) 读取
"""

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
    存储一期预测结果，自动清理超过2期的旧数据。

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

    # 仅保留最近2期，按期号排序后取最后2条
    predictions.sort(key=lambda x: x.get("period", "0"))
    predictions = predictions[-2:]

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
    """列出知识库中所有已保存的期号"""
    predictions = load_all()
    return [p.get("period", "?") for p in predictions]


def show_store():
    """展示存储文件路径和内容摘要"""
    predictions = load_all()
    return {
        "path": STORE_PATH,
        "count": len(predictions),
        "periods": [p.get("period", "?") for p in predictions],
    }
