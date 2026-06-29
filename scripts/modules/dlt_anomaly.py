#!/usr/bin/env python3
"""
DLT 数据异常检测模块

从 DLTFusionComplete 中提取的方法：
- _detect_anomalies() → detect()
- _respond_to_anomalies() → respond()
- _detect_zone_drift() → detect_zone_drift()

所有 self.xxx 引用改为 self.master.xxx
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict


class DLTAnomalyDetector:
    """DLT 数据异常检测和响应 — 异常检测 + 区间漂移检测 """

    def __init__(self, master):
        self.master = master  # DLTFusionComplete 实例引用

    def detect(self, draws: List[Tuple[List[int], List[int]]]) -> dict:
        """
        【方向E】数据异常检测 — 在模型训练前清洗数据

        检测项目:
        1. 和值Z-score: 偏离均值超过3σ的期标记为异常
        2. 号码频率卡方检验: 号码分布是否异常偏斜
        3. 重复期检测: 与之前N期完全重复

        Returns:
            {'anomaly_indices': set, 'report': str}
        """
        n = len(draws)
        if n < 20:
            return {'anomaly_indices': set(), 'report': '数据不足(need 20)'}

        anomaly_set = set()

        # --- 1. 和值Z-score异常 ---
        sums = np.array([sum(d[0]) for d in draws])
        mean_s, std_s = np.mean(sums), np.std(sums)
        z_scores = np.abs((sums - mean_s) / max(std_s, 1))

        z_anomalies = set(np.where(z_scores > 3.0)[0].tolist())
        anomaly_set.update(z_anomalies)

        # --- 2. 号码频率卡方检验 ---
        front_counter = Counter()
        for d in draws:
            front_counter.update(d[0])
        all_nums = [front_counter.get(n, 0) for n in range(1, 36)]
        expected = n * 5 / 35
        chi2 = sum((obs - expected)**2 / max(expected, 1) for obs in all_nums)
        if chi2 > 80:
            for i in range(max(0, n - 5), n):
                anomaly_set.add(i)
            anomaly_set.update(z_anomalies)

        # --- 3. 完全重复期检测 ---
        seen_front_sets = {}
        for i, d in enumerate(draws):
            fs = frozenset(d[0])
            if fs in seen_front_sets:
                prev_i = seen_front_sets[fs]
                if i - prev_i <= 50:
                    anomaly_set.add(i)
                    anomaly_set.add(prev_i)
            else:
                seen_front_sets[fs] = i

        # --- 统计 ---
        total_anomalies = len(anomaly_set)
        z_count = len(z_anomalies)
        dup_count = anomaly_set - z_anomalies
        dup_count = sum(1 for _ in dup_count) if len(anomaly_set) > 0 else 0

        report_parts = []
        if z_count > 0:
            report_parts.append(f"和值Z-score>3σ: {z_count}期")
        if chi2 > 80:
            report_parts.append(f"号码分布偏斜(χ²={chi2:.1f})")
        if dup_count > 0:
            report_parts.append(f"重复期: {dup_count}期")

        report = ', '.join(report_parts) if report_parts else '无异常'
        print(f"[DLT-Fusion] 🧹 【E】异常检测: {report}")

        return {
            'anomaly_indices': anomaly_set,
            'z_score_anomalies': len(z_anomalies),
            'chi2_stat': round(chi2, 1),
            'duplicates': dup_count,
            'total': total_anomalies,
            'report': report,
        }

    def respond(self):
        """
        【缺口D】异常检测响应回路 — 根据检测结果自动调整策略

        触发策略:
        - 轻微(单期Z-score偏高): 标记此次训练采样降权
        - 中等(连续3期异常): 主动降低延续性评分权重，降低热号权重
        - 严重(异常率>30%): 冻结预测输出，日志告警
        """
        master = self.master
        if not hasattr(master, '_anomaly_report'):
            return

        report = master._anomaly_report
        total = report.get('total', 0)
        n_draws = len(master.draws)
        if n_draws == 0:
            return

        rate = total / n_draws
        if rate < 0.05:
            return  # 正常范围，不做任何调整

        decayer = None
        try:
            from modules.ranking_feature_extractor import get_feature_decayer
            decayer = get_feature_decayer()
        except Exception:
            pass

        if rate < 0.15:
            print(f"[DLT-Fusion] ⚠️ 【D】异常率={rate:.1%}: 降延续性+增冷号权重")
            if decayer is not None:
                decayer.sleeping.update([9, 40, 41, 42, 48])
        elif rate >= 0.15:
            print(f"[DLT-Fusion] 🚨 【D】异常率={rate:.1%} > 15%: 冻结延续性评分")
            if decayer is not None:
                decayer.sleeping.update(range(40, 50))

        master._anomaly_response = {
            'trigger_rate': rate,
            'responded': True,
            'action': 'moderate' if rate < 0.15 else 'severe',
        }

    def detect_zone_drift(self, window: int = 10, drift_threshold: float = 0.15) -> Dict[str, Any]:
        """
        检测区间漂移趋势，返回各区间权重调整系数。

        核心逻辑：
        - 分析最近 window 期每期的三区分布
        - 计算"区间重心"：gravity = (1×z1_count + 2×z2_count + 3×z3_count) / 5
          重心值范围 1.0(全一区)～3.0(全三区)
        - 检测最近几期重心是否持续偏向上行或下行
        - 连续 3+ 期间方向一致 → 判定为漂移，生成区间权重偏移

        Returns:
            Dict: {
                'drift_detected': bool,
                'direction': str,
                'gravity_trend': List[float],
                'zone_adjustments': Dict[str, float],
                'confidence': float
            }
        """
        master = self.master
        draws = master.draws

        if len(draws) < window + 5:
            return {
                'drift_detected': False,
                'direction': 'stable',
                'gravity_trend': [],
                'zone_adjustments': {'z1': 1.0, 'z2': 1.0, 'z3': 1.0},
                'confidence': 0.0
            }

        recent = draws[-window:]

        ZONE_CENTER = {1: 1, 2: 2, 3: 3}
        ZONE1 = set(range(1, 13))
        ZONE2 = set(range(13, 25))
        ZONE3 = set(range(25, 36))

        gravities = []
        zone_counts = []

        for front, _ in recent:
            s = set(front)
            z1c = len(s & ZONE1)
            z2c = len(s & ZONE2)
            z3c = len(s & ZONE3)
            zone_counts.append((z1c, z2c, z3c))
            gravity = (z1c * ZONE_CENTER[1] + z2c * ZONE_CENTER[2] + z3c * ZONE_CENTER[3]) / 5.0
            gravities.append(gravity)

        if len(gravities) < 5:
            adjustments = {'z1': 1.0, 'z2': 1.0, 'z3': 1.0}
            return {
                'drift_detected': False, 'direction': 'stable',
                'gravity_trend': gravities, 'zone_adjustments': adjustments,
                'confidence': 0.0
            }

        recent_g = gravities[-5:]
        diffs = [recent_g[i+1] - recent_g[i] for i in range(len(recent_g)-1)]

        up_count = sum(1 for d in diffs if d > 0)
        down_count = sum(1 for d in diffs if d < 0)

        direction = 'stable'
        drift_detected = False
        confidence = 0.0

        if up_count >= 3 and up_count > down_count:
            direction = 'up'
            drift_detected = True
            confidence = min(up_count / 4.0, 1.0)
        elif down_count >= 3 and down_count > up_count:
            direction = 'down'
            drift_detected = True
            confidence = min(down_count / 4.0, 1.0)

        latest_z1, latest_z2, latest_z3 = zone_counts[-1]

        adjustments = {'z1': 1.0, 'z2': 1.0, 'z3': 1.0}

        if drift_detected and confidence >= drift_threshold:
            drift_strength = confidence

            if direction == 'up':
                adjustments['z1'] = max(0.4, 1.0 - drift_strength * 0.5)
                adjustments['z2'] = 1.0
                adjustments['z3'] = min(1.8, 1.0 + drift_strength * 0.6)
            elif direction == 'down':
                adjustments['z1'] = min(1.8, 1.0 + drift_strength * 0.6)
                adjustments['z2'] = 1.0
                adjustments['z3'] = max(0.4, 1.0 - drift_strength * 0.5)

            if confidence > 0.3:
                print(f"[DLT-Drift] 📊 区间漂移检测: {direction} | "
                      f"置信度={confidence:.2f} | "
                      f"调整: 一区×{adjustments['z1']:.2f} "
                      f"二区×{adjustments['z2']:.2f} "
                      f"三区×{adjustments['z3']:.2f}")

        return {
            'drift_detected': drift_detected,
            'direction': direction,
            'gravity_trend': gravities,
            'zone_adjustments': adjustments,
            'confidence': confidence
        }
