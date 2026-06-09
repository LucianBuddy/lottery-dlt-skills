#!/usr/bin/env python3
"""
DLT多策略融合完全体 V3.0.0
整合策略融合引擎 + 六池采样 + 后区融合 + 博弈论 + 遗传算法 + 数学过滤 + 统计分析
+ 隔期重号增强(SkipRepeatBooster) + 双期参考候选 + 后区隔期重号 + 智能重号惩罚 + 趋势池+尾号检测+AC值+偏差校准
"""

import sys
import os
import os.path as _path
import json
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict

# ============================================================
# 版本与参考文件同步
# ============================================================
VERSION = "3.0.2"
RELEASE_DATE = "2026-06-09"


def check_reference_sync():
    """
    检查 references/ 下的配置文件版本是否与当前代码版本一致。
    每次版本升级后应手动更新 references/ 文件中的版本号。
    返回 (synced: bool, message: str)
    """
    ref_path = _path.join(_path.dirname(_path.abspath(__file__)), '..', 'references', 'dlt_skill_config.json')
    if not _path.exists(ref_path):
        return False, f"⚠️ 参考配置文件不存在: {ref_path}"
    try:
        with open(ref_path, 'r') as f:
            config = json.load(f)
        ref_ver = config.get('reference_sync_version', '')
        if ref_ver != VERSION:
            return False, f"⚠️ 版本不匹配: 代码 VERSION={VERSION}, 配置 reference_sync_version={ref_ver}. 请更新 references/ 文件"
        return True, f"✅ references 同步正常 (V{ref_ver})"
    except Exception as e:
        return False, f"⚠️ 无法读取参考配置文件: {e}"

# 数据文件路径（基于技能包目录自动定位）
def data_dir() -> str:
    return _path.join(_path.dirname(_path.abspath(__file__)), '..', 'assets', 'data', 'DLT历史数据_适配模型版.xlsx')


# 导入所有子模块
from dlt_predictor_upgraded import DLTPredictorUpgraded, load_dlt_data
from strategy_fusion_engine import StrategyFusionEngine
from five_pool_sampler_complete_final import MultiPoolSampler
from dlt_back_fusion import BackZoneFusion
from modules.dlt_game_theory import DLTGameTheoryAnalyzer
from modules.dlt_genetic_optimizer import DLTGeneticOptimizer
from modules.dlt_math_filter import DLTMathFilter
from modules.dlt_statistics_analyzer import DLTStatisticsAnalyzer
from modules.dlt_pattern_recognizer import DLTPatternRecognizer, apply_pattern_boost, generate_pattern_diversity_pool
from modules.neural_models import NeuralEnsemble
from modules.neural_models import TORCH_AVAILABLE as TORCH_OK
from dlt_constraint_engine import DLTConstraintEngine
from modules.dlt_compound_betting import generate_all_compound, filter_and_score, select_diverse


class DLTFusionComplete:
    """DLT多策略融合完全体 — 整合所有预测模块的统一入口"""

    def __init__(self, data_path: Optional[str] = None, auto_update: bool = True):
        if data_path is None:
            data_path = data_dir()
        self.data_path = data_path

        # 1. 加载数据（带fallback）
        self._periods = []  # 期号列表，由_load_data填充
        self.draws = self._load_data(data_path)
        if not self.draws:
            raise ValueError(f"数据加载失败: {data_path}")

        print(f"[DLT-Fusion] 历史数据加载完成 | 共{len(self.draws)}期")

        # 2. 初始化所有子模块
        try:
            self.predictor = DLTPredictorUpgraded(data_path)
        except Exception as e:
            print(f"[DLT-Fusion] DLTPredictorUpgraded初始化失败: {e}")
            self.predictor = None

        self.sfe = StrategyFusionEngine(self.draws, n_groups=5)
        self.pool_sampler = MultiPoolSampler(self.draws)
        self.back_fusion = BackZoneFusion(self.draws)
        self.game_theory = DLTGameTheoryAnalyzer()
        self.genetic = DLTGeneticOptimizer(self.draws)
        self.math_filter = DLTMathFilter()
        self.stats = DLTStatisticsAnalyzer(self.draws)

        # 6. 初始化跨期模式识别器
        self.pattern_recognizer = DLTPatternRecognizer(self.draws)
        try:
            self.pattern_recognizer.build_distributions(window=500)
            print(f"[DLT-Fusion] 🧩 跨期模式识别器初始化完成 (500期分布)")
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 模式识别器初始化失败: {e}")

        # 7. 初始化神经网络模型（TabNet + LSTM + Transformer）
        self.neural_ensemble = None
        try:
            self.neural_ensemble = NeuralEnsemble(
                self.draws, seq_len=20, window=50,
                train_epochs=50, auto_train=True
            )
            if TORCH_OK and self.neural_ensemble.is_trained:
                print(f"[DLT-Fusion] 🧠 神经网络集成器初始化完成 (TabNet+LSTM+Transformer)")
            else:
                print(f"[DLT-Fusion] 🧠 神经网络集成器已加载 (fallback模式)")
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 神经网络初始化跳过: {e}")

        # 初始化遗传算法
        try:
            self.genetic.evolve(generations=50, verbose=False)
        except Exception as e:
            print(f"[DLT-Fusion] 遗传算法初始化: {e}")

        print(f"[DLT-Fusion] 初始化完成 | V3.0.2 + NeuralEnsemble")

    def _load_data(self, path: str) -> List[Tuple[List[int], List[int]]]:
        """加载数据，支持多种fallback"""
        # 优先使用dlt_predictor_upgraded的加载函数
        try:
            if os.path.exists(path):
                draws = load_dlt_data(path)
                if draws:
                    print(f"[DLT-Fusion] 从文件加载: {len(draws)}期")
                    return draws
        except Exception as e:
            print(f"[DLT-Fusion] load_dlt_data失败: {e}")

        # 尝试直接读取Excel
        try:
            df = pd.read_excel(path)
            draws = []
            # 读取期号列
            self._periods = []
            for j in range(len(df)):
                front = sorted([int(df.iloc[j][f'前区{i}']) for i in range(1, 6)])
                back = sorted([int(df.iloc[j][f'后区{i}']) for i in range(1, 3)])
                draws.append((front, back))
                if '期号' in df.columns:
                    self._periods.append(int(df.iloc[j]['期号']))
            if draws and draws[0][0][0] > draws[-1][0][0]:
                draws = list(reversed(draws))
                if self._periods:
                    self._periods = list(reversed(self._periods))
            if draws:
                print(f"[DLT-Fusion] 直接读取Excel: {len(draws)}期")
                if self._periods:
                    print(f"[DLT-Fusion] 期号范围: {self._periods[0]}~{self._periods[-1]}")
                return draws
        except Exception as e:
            print(f"[DLT-Fusion] 直接读取Excel失败: {e}")

        return []

    # ------------------------------------------------------------------
    # 区间漂移检测器 (Zone Drift Detector)
    # ------------------------------------------------------------------

    def _detect_zone_drift(self, window: int = 10, drift_threshold: float = 0.15) -> Dict[str, Any]:
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
                'drift_detected': bool,      # 是否检测到漂移
                'direction': str,             # 'up'(向高区) / 'down'(向低区) / 'stable'
                'gravity_trend': List[float], # 最近每期重心值
                'zone_adjustments': Dict[str, float],  # {z1: 1.0, z2: 1.0, z3: 1.0} 权重系数
                'confidence': float           # 0~1, 漂移置信度
            }
        """
        if len(self.draws) < window + 5:
            return {
                'drift_detected': False,
                'direction': 'stable',
                'gravity_trend': [],
                'zone_adjustments': {'z1': 1.0, 'z2': 1.0, 'z3': 1.0},
                'confidence': 0.0
            }

        recent = self.draws[-window:]

        # 定义各区中心权重（用于计算重心）
        ZONE_CENTER = {1: 1, 2: 2, 3: 3}  # 一区=1, 二区=2, 三区=3
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
            # 重心 = 加权平均
            gravity = (z1c * ZONE_CENTER[1] + z2c * ZONE_CENTER[2] + z3c * ZONE_CENTER[3]) / 5.0
            gravities.append(gravity)

        # 检测漂移方向：连续 N 期重心持续上升或下降
        if len(gravities) < 5:
            adjustments = {'z1': 1.0, 'z2': 1.0, 'z3': 1.0}
            return {
                'drift_detected': False, 'direction': 'stable',
                'gravity_trend': gravities, 'zone_adjustments': adjustments,
                'confidence': 0.0
            }

        # 取最近 5 期的重心变化
        recent_g = gravities[-5:]
        diffs = [recent_g[i+1] - recent_g[i] for i in range(len(recent_g)-1)]

        # 判断漂移：连续同向变化
        up_count = sum(1 for d in diffs if d > 0)
        down_count = sum(1 for d in diffs if d < 0)

        # 决定方向
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

        # 计算本期实际区间分布
        latest_z1, latest_z2, latest_z3 = zone_counts[-1]

        # 计算调整系数
        adjustments = {'z1': 1.0, 'z2': 1.0, 'z3': 1.0}

        if drift_detected and confidence >= drift_threshold:
            # 漂移强度 = 置信度 × 漂移方向
            # 【优化V3.0.2】提高调整幅度，使漂移检测更显著影响候选分布
            drift_strength = confidence

            if direction == 'up':
                # 重心在向高区移动：提升三区权重，降低一区权重（幅度增大33%）
                adjustments['z1'] = max(0.4, 1.0 - drift_strength * 0.5)
                adjustments['z2'] = 1.0  # 二区不变
                adjustments['z3'] = min(1.8, 1.0 + drift_strength * 0.6)
            elif direction == 'down':
                # 重心在向低区移动：提升一区权重，降低三区权重（幅度增大33%）
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

    def _apply_drift_boost(self, candidates: List[Dict[str, Any]],
                            drift_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        区间漂移补偿：根据漂移方向调整候选评分。

        对候选的前区号码计算区间匹配度，
        匹配漂移方向的候选获得加分，不匹配的降分。
        """
        ZONE1 = set(range(1, 13))
        ZONE2 = set(range(13, 25))
        ZONE3 = set(range(25, 36))

        direction = drift_info['direction']
        adjustments = drift_info['zone_adjustments']
        confidence = drift_info['confidence']

        boost_strength = min(confidence * 0.20, 0.15)  # max 15% score adjustment（V3.0.2 提高）

        boosted_count = 0
        penalized_count = 0

        for c in candidates:
            front = set(c.get('front', []))
            if not front:
                continue

            z1c = len(front & ZONE1)
            z2c = len(front & ZONE2)
            z3c = len(front & ZONE3)

            # 计算该候选的"匹配漂移方向"得分
            if direction == 'up':
                # 向高区漂移：奖励三区占比高的候选
                zone_fit = (z3c * adjustments['z3'] + z2c * adjustments['z2']
                           - z1c * (1.0 / adjustments['z1'] if adjustments['z1'] > 0 else 1.0)) / 5.0
            elif direction == 'down':
                # 向低区漂移：奖励一区占比高的候选
                zone_fit = (z1c * adjustments['z1'] + z2c * adjustments['z2']
                           - z3c * (1.0 / adjustments['z3'] if adjustments['z3'] > 0 else 1.0)) / 5.0
            else:
                zone_fit = 0.0

            # 限制到 [-0.5, 0.5] 范围
            zone_fit = max(-0.5, min(0.5, zone_fit))

            # 应用评分调整：加分或减分
            orig = c.get('final_score', c.get('base_score', 0.5))
            adjustment = zone_fit * boost_strength
            c['final_score'] = max(0.1, orig + adjustment)
            c['drift_adjustment'] = adjustment

            if adjustment > 0:
                boosted_count += 1
            elif adjustment < 0:
                penalized_count += 1

        if boosted_count > 0 or penalized_count > 0:
            print(f"[DLT-Drift] 🔼区间漂移补偿: +{boosted_count}注加分 / {penalized_count}注减分 "
                  f"(强度={boost_strength:.4f})")

        return candidates

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def _reinit_submodules(self):
        """数据更新后重新初始化所有子模块"""
        try:
            self.sfe = StrategyFusionEngine(self.draws, n_groups=5)
            self.pool_sampler = MultiPoolSampler(self.draws)
            self.back_fusion = BackZoneFusion(self.draws)
            self.genetic = DLTGeneticOptimizer(self.draws)
            self.stats = DLTStatisticsAnalyzer(self.draws)
            self.pattern_recognizer = DLTPatternRecognizer(self.draws)
            self.pattern_recognizer.build_distributions(window=500)
            self.genetic.evolve(generations=50, verbose=False)
            # 约束引擎重新初始化
            try:
                self.constraint_engine = DLTConstraintEngine()
            except Exception:
                pass
            # 神经网络重新训练
            try:
                self.neural_ensemble = NeuralEnsemble(
                    self.draws, seq_len=20, window=50,
                    train_epochs=30, auto_train=True
                )
                if TORCH_OK and self.neural_ensemble.is_trained:
                    print(f"[DLT-Fusion] 🔄 神经网络模型已重新训练")
            except Exception as e:
                print(f"[DLT-Fusion] ⚠️ 神经网络重训练跳过: {e}")

            print(f"[DLT-Fusion] 🔄 子模块已基于新数据重新初始化 ({len(self.draws)}期)")
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 子模块重初始化失败: {e}")

    def get_group_recommendations(self) -> Dict[int, List[Any]]:
        """获取5组独立推荐"""
        results = self.sfe.generate_all_groups(n_per_group=5)
        return results

    def get_back_recommendations(self) -> List[List[int]]:
        """获取后区推荐（返回后区号码对列表）"""
        raw = self.back_fusion.generate_recommendations(n=5)
        # raw = List[Tuple[List[int], Dict[str,float]]]
        return [back_pair for back_pair, _ in raw]

    def generate_compound_bets(self, bet_type: str = '6+3', n_per_type: int = 2) -> Dict[str, List[Dict]]:
        """
        生成多种复式投注（V2.2 — 全量枚举 + 数学过滤 + 多样性选择 + 评分集成）

        流程:
          1. 从多池策略合成前/后区候选池（而非单策略硬编码）
          2. generate_all_compound() 全量枚举 C(N,5)×C(M,2) 组合
          3. filter_and_score() 数学约束过滤（和值/奇偶/AC值）
          4. 注入 final_score（继承单式预测评分体系）
          5. select_diverse() 贪心多样性选择

        Args:
            bet_type: '6+3','6+4','7+2','7+3','7+4','8+2','8+3','8+4','8+5',
                     '9+3','9+4','9+6','all'
            n_per_type: 每种类型输出多少组

        Returns:
            Dict[bet_type, List[Dict]]
        """
        compound_types = [
            ('6+3', 6, 3), ('6+4', 6, 4),
            ('7+2', 7, 2), ('7+3', 7, 3), ('7+4', 7, 4),
            ('8+2', 8, 2), ('8+3', 8, 3), ('8+4', 8, 4), ('8+5', 8, 5),
            ('9+3', 9, 3), ('9+4', 9, 4), ('9+6', 9, 6),
        ]

        if bet_type != 'all':
            matched = [(l, fc, bc) for l, fc, bc in compound_types if l == bet_type]
            if not matched:
                return {bet_type: []}
            compound_types = matched

        results = {}
        for label, fc, bc in compound_types:
            try:
                # Step 1: 从多池获取扩展候选池
                front_pool = self._get_compound_front_pool(fc + 8)
                back_pool = self._get_compound_back_pool(bc + 3)

                # Step 2: 枚举 fc 个前区中选 fc 个的不同组合作为完整候选池
                # 取前 fc*2 个号码，从中挑选 fc 个的不同子集作为不同方案
                from itertools import combinations
                pool_front_candidates = list(combinations(front_pool[:fc * 2], fc))
                pool_back_candidates = list(combinations(back_pool[:bc * 2], bc))

                # 对每种候选池组合评分
                scored_pools = []
                for f_pool in pool_front_candidates:
                    for b_pool in pool_back_candidates:
                        # 用候选池中所有5+2组合的平均分作为该池的评分
                        all_5_2 = generate_all_compound(list(f_pool), list(b_pool), fc, bc)
                        pool_score = 0.0
                        count = 0
                        for cfc, cbc in all_5_2:
                            score = self._calc_compound_score(list(cfc), list(cbc))
                            pool_score += score
                            count += 1
                        avg_score = pool_score / max(count, 1)
                        scored_pools.append((list(f_pool), list(b_pool), avg_score))

                # 按评分排序
                scored_pools.sort(key=lambda x: -x[2])

                # 多样性选择：从top候选池中选n_per_type个不同池
                selected_pools = []
                seen_front = []
                for f_pool, b_pool, scr in scored_pools:
                    # 前区重叠度不超过 fc-2 即视为不同方案
                    unique_enough = True
                    for prev_f in seen_front:
                        overlap = len(set(f_pool) & set(prev_f))
                        if overlap >= fc - 1:
                            unique_enough = False
                            break
                    if unique_enough:
                        selected_pools.append((f_pool, b_pool, scr))
                        seen_front.append(f_pool)
                        if len(selected_pools) >= n_per_type:
                            break

                # 如果没有多样性候选，直接取top-n
                if len(selected_pools) < n_per_type:
                    for f_pool, b_pool, scr in scored_pools:
                        if (f_pool, b_pool) not in [(s[0], s[1]) for s in selected_pools]:
                            selected_pools.append((f_pool, b_pool, scr))
                            if len(selected_pools) >= n_per_type:
                                break

                result_list = []
                for f_pool, b_pool, scr in selected_pools[:n_per_type]:
                    result_list.append({
                        'front_pool': sorted(f_pool),   # 完整前区候选池（如6个号）
                        'back_pool': sorted(b_pool),    # 完整后区候选池（如3个号）
                        'front_count': fc,
                        'back_count': bc,
                        'pool_score': round(scr, 4),
                        'bet_type': label,
                    })

                results[label] = result_list
            except Exception as e:
                print(f"[DLT-Fusion] ⚠️ 复式 {label} 生成失败: {e}")
                results[label] = []

        return results

    def _get_compound_front_pool(self, target_size: int = 14) -> List[int]:
        """
        从6池加权采样合成复式前区候选池。
        【优化V3.0.2】与单注采样路径分离：
        - 使用不同的expand比率（采更多号码确保覆盖）
        - 去掉上期排除过滤（保留合理重号）
        - 强制性确保三区都有代表号码
        - 从热/冷/均衡/趋势池中采更多以增加多样性
        """
        # 扩展池大小取 target_size 的 2 倍，确保有充足候选
        extended = target_size * 2
        pool = []
        pool.extend(self.pool_sampler.generate_hot_pool(extended, 'front'))
        pool.extend(self.pool_sampler.generate_cold_pool(extended, 'front'))
        pool.extend(self.pool_sampler.generate_balance_pool(extended, 'front'))
        pool.extend(self.pool_sampler.generate_trend_pool(extended, 'front'))
        pool.extend(self.pool_sampler.generate_game_theory_pool(extended, 'front'))
        pool.extend(self.pool_sampler.generate_genetic_pool(extended, 'front'))
        # 去重
        seen = set()
        unique = [x for x in pool if not (x in seen or seen.add(x))]

        # 三区覆盖检查：确保候选池各区间都有号码
        ZONE1 = set(range(1, 13))
        ZONE2 = set(range(13, 25))
        ZONE3 = set(range(25, 36))
        uni_set = set(unique)
        for zi, zn in [(1, ZONE1), (2, ZONE2), (3, ZONE3)]:
            if not (uni_set & zn):
                # 该区无号码，从全范围补充
                extra = [n for n in zn if n not in uni_set]
                if extra:
                    pick = max(extra, key=lambda n: abs(n - 18))
                    unique.append(pick)
                    uni_set.add(pick)

        # 保留上期号码（不排除），因为重号是合理模式
        return list(set(unique))[:target_size]

    def _get_compound_back_pool(self, target_size: int = 6) -> List[int]:
        """
        合成后区候选池。
        【优化V3.0.2】独立采样，不依赖单注池结果。
        扩展采样量确保覆盖后区所有号码。
        """
        extended = max(target_size * 2, 12)
        pool = []
        pool.extend(self.pool_sampler.generate_hot_pool(extended, 'back'))
        pool.extend(self.pool_sampler.generate_cold_pool(extended, 'back'))
        pool.extend(self.pool_sampler.generate_balance_pool(extended, 'back'))
        pool.extend(self.pool_sampler.generate_game_theory_pool(extended, 'back'))
        pool.extend(self.pool_sampler.generate_trend_pool(extended, 'back'))
        seen = set()
        unique = [x for x in pool if not (x in seen or seen.add(x))]
        # 确保后区号码充分覆盖（所有1-12都可能在候选池）
        uni_set = set(unique)
        for n in range(1, 13):
            if n not in uni_set:
                unique.append(n)
        return unique[:target_size]

    def _inject_compound_scores(self, combos):
        """
        为复合方案注入 final_score，继承单式预测评分体系。

        Args:
            combos: List of (front_tuple, back_tuple)
        Returns:
            List of (front_tuple, back_tuple, final_score)
        """
        scored = []
        for fc, bc in combos:
            score = self._calc_compound_score(list(fc), list(bc))
            scored.append((fc, bc, score))
        return scored

    def _calc_compound_score(self, front: List[int], back: List[int]) -> float:
        """
        计算一组号码的复合评分，与单式预测评分体系一致。
        综合：博弈论 + 数学特征 + 历史频率 + 后区命中率
        """
        score = 0.5  # baseline

        # 1. 博弈论评分 (25%权重)
        try:
            analysis = self.game_theory.analyze_combo(front)
            gt_score = analysis['scores']['combined_score']
            score += gt_score * 0.25
        except Exception:
            pass

        # 2. 数学特征评分 (20%)
        try:
            front_sum = sum(front)
            sum_dev = abs(front_sum - 98) / 45  # 98=理论均值
            math_score = max(0, 1 - sum_dev) * 0.20
            score += math_score
        except Exception:
            pass

        # 3. 历史频率评分 (20%) — 最近30期每个号码出现频率越高越好
        try:
            if hasattr(self, 'pool_sampler'):
                from collections import Counter
                recent_draws = self.draws[-30:] if len(self.draws) >= 30 else self.draws
                freq = Counter()
                for d in recent_draws:
                    for n in d[0]:
                        freq[n] += 1
                avg_freq = sum(freq[n] for n in front if n in freq) / max(len(front), 1)
                max_possible = max(freq.values()) if freq else 1
                freq_score = avg_freq / max_possible * 0.20
                score += freq_score
        except Exception:
            pass

        # 4. 后区命中率 (15%) — 最近后区号码热度
        try:
            if hasattr(self, 'back_fusion'):
                recent_back = []
                for d in self.draws[-20:]:
                    recent_back.extend(d[1])
                back_freq = Counter(recent_back)
                back_hits = sum(back_freq.get(n, 0) for n in back)
                back_score = back_hits / max(len(recent_back), 1) * 15
                score += back_score
        except Exception:
            pass

        # 5. 模式识别匹配度 (20%) — 号码分布是否符合当前期历史模式
        try:
            if hasattr(self, 'pattern_recognizer') and self.pattern_recognizer._is_built:
                pr = self.pattern_recognizer
                front_sum = sum(front)
                span = max(front) - min(front)
                odd_cnt = sum(1 for n in front if n % 2 == 1)
                even_cnt = 5 - odd_cnt

                match = 0.0
                # 和值模式
                s_dist = pr.pattern_stats.get('sum', {}).get('distribution', {})
                if s_dist:
                    bucket = (front_sum // 5) * 5
                    freq = s_dist.get(bucket, 0)
                    total = sum(s_dist.values()) or 1
                    match += freq / total * 0.05
                # 跨度模式
                sp_dist = pr.pattern_stats.get('span', {}).get('distribution', {})
                if sp_dist:
                    bucket = (span // 5) * 5
                    freq = sp_dist.get(bucket, 0)
                    total = sum(sp_dist.values()) or 1
                    match += freq / total * 0.05
                # 奇偶模式
                oe_dist = pr.pattern_stats.get('odd_even_ratio', {}).get('distribution', {})
                if oe_dist:
                    key = f"{odd_cnt}:{even_cnt}"
                    freq = oe_dist.get(key, 0)
                    total = sum(oe_dist.values()) or 1
                    match += freq / total * 0.05
                # AC值模式
                ac_dist = pr.pattern_stats.get('ac_value', {}).get('distribution', {})
                if ac_dist:
                    ac_val = len(set(abs(front[i] - front[j]) for i in range(5) for j in range(i + 1, 5))) - 4
                    freq = ac_dist.get(ac_val, 0)
                    total = sum(ac_dist.values()) or 1
                    match += freq / total * 0.05

                score += match * 0.20
        except Exception:
            pass

        return min(max(score, 0), 1.0)

    def generate_dantuo_bets(self, dan_front: Optional[List[int]] = None,
                            tuo_front_size: int = 8,
                            dan_back: Optional[List[int]] = None,
                            tuo_back_size: int = 4,
                            n_sets: int = 3,
                            n_dan_front: int = 2) -> List[Dict[str, Any]]:
        """
        胆拖投注生成（公共接口）

        流程:
          1. 如果用户指定胆码，直接使用；否则从各策略池自动选取高分胆码
          2. 拖码通过多池加权合成（同复式候选池逻辑）
          3. 多池合成拖码 + 全排列 + 多样性选择

        胆码个数限制:
          - 前区: 1~4 个（剩余由拖码补齐到5）
          - 后区: 0~1 个（剩余由拖码补齐到2）

        Args:
            dan_front: 用户指定的前区胆码（None=自动选）
            tuo_front_size: 前区拖码池大小（默认8）
            dan_back: 用户指定的后区胆码（None=自动选）
            tuo_back_size: 后区拖码池大小（默认4）
            n_sets: 生成组数
            n_dan_front: 自动选胆时的胆码个数 (1-4, 默认2)

        Returns:
            [{'name', 'front_dan', 'front_tuo', 'back', 'total_bets', 'hit_probability'}, ...]
        """
        results = []

        try:
            # 合成候选池（同复式逻辑）
            front_pool = self._get_compound_front_pool(tuo_front_size + 5)
            back_pool = self._get_compound_back_pool(tuo_back_size + 2)

            # 胆码处理
            if dan_front is not None:
                dan_f = sorted(set(dan_front))
                if not (1 <= len(dan_f) <= 4):
                    raise ValueError(f"前区胆码需1~4个, 当前{len(dan_f)}")
                # 从候选池移除胆码
                front_pool = [n for n in front_pool if n not in dan_f]
            else:
                # 自动选胆：取前N个高频号为胆
                n_dan_front = max(1, min(4, n_dan_front))
                freq = Counter()
                for f, _ in self.draws[-50:]:
                    freq.update(f)
                ranked = sorted(freq.keys(), key=lambda n: -freq.get(n, 0))
                dan_f = ranked[:min(n_dan_front, len(ranked))]
                front_pool = [n for n in front_pool if n not in dan_f]

            # 后区胆码
            if dan_back is not None:
                dan_b = sorted(set(dan_back))
                if not (1 <= len(dan_b) <= 1):
                    raise ValueError(f"后区胆码需0或1个, 当前{len(dan_b)}")
                back_pool = [n for n in back_pool if n not in dan_b]
            else:
                dan_b = []

            tuo_f = sorted(front_pool[:tuo_front_size])
            tuo_b = sorted(back_pool[:tuo_back_size])
            nd = len(dan_f)
            nt = 5 - nd  # 需要补充的拖码数

            if len(tuo_f) < nt:
                raise ValueError(f"拖码不足: 需要{nt}个, 只有{len(tuo_f)}")

            from itertools import combinations as _combs

            # 胆码固定，拖码全排列
            tuo_combos = list(_combs(tuo_f, nt))
            bt = 2 - len(dan_b)
            back_combos = list(_combs(tuo_b, bt)) if bt >= 1 else [()]

            total_bets = len(tuo_combos) * max(len(back_combos), 1)

            # 注入评分
            scored = []
            for tc in tuo_combos:
                full_f = sorted(dan_f + list(tc))
                for bc in back_combos:
                    full_b = sorted(dan_b + list(bc))
                    score = self._calc_compound_score(full_f, full_b)
                    scored.append((full_f, full_b, score))
            scored.sort(key=lambda x: -x[2])

            # 多样性选择
            selected_tuples = []
            for ff, fb, _ in scored[:max(n_sets * 3, len(scored))]:
                selected_tuples.append((tuple(ff), tuple(fb)))

            from modules.dlt_compound_betting import select_diverse as _sd
            diverse = _sd(selected_tuples, n_sets)

            # 多样性方案：每次选不同的子组合，但展示完整拖池
            # （前区拖池固定，多样性的价值在后区配对）
            for ff, fb in diverse:
                probs = self._calc_probability(list(ff), list(fb))
                # 计算后区完整候选池（含胆码）
                back_pool_full = sorted(dan_b + tuo_b) if dan_b else sorted(tuo_b)
                results.append({
                    'name': f"{nd}胆{nt}拖",
                    'front_dan': dan_f,
                    'front_tuo_pool': tuo_f,              # 完整拖池（如8个号）
                    'back': list(fb),                     # 推荐的1组后区（多样性）
                    'back_pool_full': back_pool_full,     # 后区完整候选池
                    'total_bets': total_bets,
                    'hit_probability': probs['combined'],
                })

        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 胆拖投注生成失败: {e}")

        results.sort(key=lambda x: -x['hit_probability'])
        return results[:n_sets]

    def recommend_stake(self, budget: float = 100.0,
                        predictions: Optional[List[Dict[str, Any]]] = None,
                        half_kelly: bool = True) -> List[Dict[str, Any]]:
        """
        凯利公式投注建议

        用 half-Kelly 计算每注建议投注额:
          f* = (p*(b+1)-1)/b
        其中 p=命中概率, b=赔率(大乐透固定头奖约89万倍)

        Args:
            budget: 总预算（元）
            predictions: 预测结果列表，每项含 hit_probability
                         默认从最近一次 predict() 的 single_bets 取
            half_kelly: 是否使用 half-Kelly（默认 True，更保守）

        Returns:
            [{'bet', 'front', 'back', 'hit_prob', 'kelly_pct', 'stake_yuan', 'note'}, ...]
        """
        if predictions is None:
            # 尝试直接预测一次
            try:
                pred = self.predict(top_n=5, include_compound=False)
                predictions = pred.get('single_bets', [])
            except Exception:
                return []

        if not predictions:
            return []

        # 大乐透头奖赔率估算：2元中1000万（固定奖）
        # 实际上还有各级小奖，简化处理：b = 5,000,000
        b = 5_000_000.0

        stakes = []
        total_kelly = 0.0

        for pred in predictions:
            p = pred.get('hit_probability', 0.0)
            if p <= 0 or p >= 1:
                continue

            kelly = (p * (b + 1) - 1) / b
            if kelly < 0:
                kelly = 0.0

            if half_kelly:
                kelly /= 2.0

            total_kelly += kelly

            stakes.append({
                'bet': pred.get('bet', ''),
                'front': pred.get('front', []),
                'back': pred.get('back', []),
                'hit_prob': round(p * 100, 2),
                'kelly_pct': round(kelly * 100, 4),
                'stake_yuan': 0.0,
                'note': '不建议投注' if kelly <= 0 else '',
            })

        # 归一化到预算
        if total_kelly > 0:
            for s in stakes:
                s['stake_yuan'] = round(budget * (s['kelly_pct'] / 100) / total_kelly, 2)

        return stakes

    def predict(self, top_n: int = 5, include_compound: bool = True) -> Dict[str, Any]:
        """主预测函数：触发预测时自动同步最新开奖数据，然后生成推荐"""
        # Step -1: 版本同步检查（仅warning级别，不影响预测）
        synced, sync_msg = check_reference_sync()
        if not synced:
            print(sync_msg)

        # Step 0: 触发预测时同步最新开奖数据
        try:
            from dlt_data_updater import check_and_update
            update_result = check_and_update()
            if update_result['updated']:
                print(f"[DLT-Fusion] 📥 已同步新数据: +{update_result['new_count']}期 "
                      f"({update_result['new_periods'][0]}~{update_result['new_periods'][-1]})")
                # 数据有更新，重新加载子模块
                self.draws = self._load_data(self.data_path)
                self._reinit_submodules()
            else:
                print(f"[DLT-Fusion] ✅ 数据已是最新 (最新期号: {update_result['last_period']})")
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 数据更新检查跳过: {e}")

        # Step 0.5: 区间漂移检测 — 当检测到漂移时，候选中将追加区间漂移补偿候选
        drift_info = self._detect_zone_drift()
        zone_adj = drift_info['zone_adjustments']

        # Step 1: SFE 5组融合
        all_groups = self.get_group_recommendations()

        # Step 2: 多池采样补充候选 + 模式池采样 + 双期重号参考候选
        pool_candidates = self._sample_pool_candidates(n_per_pool=2)
        try:
            pattern_candidates = self._sample_pattern_pool_candidates()
            pool_candidates.extend(pattern_candidates)
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 模式池采样跳过: {e}")

        # 【方案B】双期重号参考候选池 — 覆盖隔期号码回归模式
        try:
            skip_repeat_candidates = self._sample_skip_repeat_candidates()
            pool_candidates.extend(skip_repeat_candidates)
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 双期参考候选项跳过: {e}")

        # Step 3: 后区融合
        back_recs = self.get_back_recommendations()

        # Step 4: 博弈论优化
        gt_scores = self._apply_game_theory(all_groups, pool_candidates)

        # Step 5: 遗传算法优化
        genetic_scores = self._apply_genetic_optimization(all_groups, pool_candidates)

        # Step 6: 汇总所有候选
        all_candidates = self._collect_candidates(all_groups, pool_candidates, gt_scores, genetic_scores)

        # Step 6a: 🎯 奇偶比分布补充 — 确保候选覆盖所有常见奇偶比模式
        try:
            all_candidates = self._enrich_parity_distribution(all_candidates)
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 奇偶比补充跳过: {e}")

        # Step 6b: 📊 和值中间带补充（P4）— 当候选集和值分布缺乏中间带(100-120)时补偿
        try:
            all_candidates = self._compensate_mid_sum(all_candidates)
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 和值补偿跳过: {e}")

        # Step 7a: 常规综合评分 (base*0.4 + gt*0.3 + genetic*0.3)
        self._compute_final_scores(all_candidates)

        # Step 7b: 跨期模式评分增强（方案二：叠加模式匹配度评分）
        if hasattr(self, 'pattern_recognizer') and self.pattern_recognizer._is_built:
            prev_front = self.draws[-1][0] if len(self.draws) >= 1 else None
            all_candidates = apply_pattern_boost(
                all_candidates, self.pattern_recognizer,
                prev_front=prev_front, boost_weight=0.35
            )

        # Step 7b2: 区间漂移补偿 — 调整候选评分，偏向漂移方向
        # 【优化V3.0.2】降低漂移检测启动阈值 0.15→0.10
        if drift_info['drift_detected'] and drift_info['confidence'] >= 0.10:
            all_candidates = self._apply_drift_boost(all_candidates, drift_info)

        # 【方案A】隔期重号评分增强 — 候选与上上期匹配时加分
        try:
            all_candidates = self._apply_skip_repeat_boost(all_candidates)
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 隔期重号评分跳过: {e}")

        # Step 7c: 重号惩罚——候选与上期重号≥3个时降分5%，避免热号过度堆叠
        all_candidates = self._apply_repeat_penalty(all_candidates)

        # Step 7c2: 特征工程——区间分布均衡评分 + 散度特征评分 + 尾号聚合检测 + AC值跟踪评分
        all_candidates = self._apply_zone_balance_scoring(all_candidates)
        all_candidates = self._apply_scatter_scoring(all_candidates)
        all_candidates = self._apply_tail_density_scoring(all_candidates)
        all_candidates = self._apply_ac_value_scoring(all_candidates)

        # Step 7c2.5: 🧠 神经网络评分（TabNet + LSTM + Transformer）
        try:
            if self.neural_ensemble is not None and self.neural_ensemble.is_trained:
                self.neural_ensemble.score_batch(all_candidates, self.draws)
                # 对已有neural_score的候选做混合
                for c in all_candidates:
                    ns = c.get('neural_score', 0.5)
                    c['final_score'] = c['final_score'] * 0.75 + ns * 0.25
                print(f"[DLT-Fusion] 🧠 神经网络评分已集成 (+25%权重)")
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 神经网络评分跳过: {e}")

        # Step 7c2.6: 🎯 号码过度集中抑制（P1）— 防止候选集被少数热号垄断
        try:
            all_candidates = self._apply_diversity_penalty(all_candidates)
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 多样性惩罚跳过: {e}")

        # Step 7c3: 偏差仪表盘 + 置信度重校准
        dashboard = self._compute_deviation_dashboard(all_candidates)
        all_candidates = self._recalibrate_confidence(all_candidates, dashboard)

        # Step 7c4: 策略约束验证 — 用6种策略模板过滤候选，至少通过2种
        try:
            if hasattr(self, 'constraint_engine') and self.constraint_engine is not None:
                valid = []
                for c in all_candidates:
                    front = c.get('front', [])
                    back = c.get('back', [1, 12])
                    ok, _ = self.constraint_engine.validate_hard(front, back)
                    if not ok:
                        continue
                    # 统计通过了几种策略
                    pass_cnt = 0
                    for st in range(1, 7):
                        sk, _ = self.constraint_engine.validate_strategy(front, back, st)
                        if sk:
                            pass_cnt += 1
                    c['strategy_pass_count'] = pass_cnt
                    if pass_cnt >= 2:
                        valid.append(c)
                if valid:
                    kept = len(valid)
                    all_candidates = valid
                    print(f"[DLT-Fusion] 🔗 策略约束过滤: {kept} 候选通过 (原始{len(all_candidates)})")
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 策略约束验证跳过: {e}")

        # Step 7d: 过滤掉与最近一期完全相同的号码（不可能连续两期一模一样）
        all_candidates = self._filter_recent_draws(all_candidates)

        # Step 8: 去重+分配后区
        unique = self._deduplicate_and_assign_back(all_candidates, back_recs)

        # Step 9: 防御性去重——确保每注内的前区和后区号码唯一
        for bet in unique:
            # 前区去重排序
            bet['front'] = sorted(set(bet['front']))
            # 后区去重排序
            bet['back'] = sorted(set(bet['back']))
            # 补足到5个前区号码（万一set去重后少了）
            while len(bet['front']) < 5:
                fill = random.randint(1, 35)
                if fill not in bet['front']:
                    bet['front'].append(fill)
            bet['front'].sort()
            # 补足到2个后区号码
            while len(bet['back']) < 2:
                fill = random.randint(1, 12)
                if fill not in bet['back']:
                    bet['back'].append(fill)
            bet['back'].sort()

        # Step 10: 最终排名并计算命中概率
        unique.sort(key=lambda x: x['final_score'], reverse=True)
        for bet in unique:
            try:
                probs = self._calc_probability(bet['front'], bet['back'])
                bet['hit_probability'] = probs['combined']
                bet['front_prob'] = probs['front']
                bet['back_prob'] = probs['back']
            except Exception:
                bet['hit_probability'] = 0.0
                bet['front_prob'] = 0.0
                bet['back_prob'] = 0.0

        result = {
            'single_bets': unique[:top_n],
        }

        # 添加复式投注推荐
        if include_compound:
            compound = self.generate_compound_bets('all', n_per_type=2)
            result['compound_bets'] = compound

            # 添加胆拖投注方案（多池融合版）
            try:
                all_dantuo = []
                # 1胆/2胆/3胆 三种配置，拖码池大小逐步缩小
                configs = [(1, 10), (2, 8), (3, 6)]
                for ndan, tsize in configs:
                    schemes = self.generate_dantuo_bets(
                        dan_front=None,
                        tuo_front_size=tsize,
                        n_sets=2,
                        n_dan_front=ndan,
                    )
                    for s in schemes:
                        if s not in all_dantuo:
                            all_dantuo.append(s)

                if all_dantuo:
                    all_dantuo.sort(key=lambda x: -x['hit_probability'])
                    result['dan_tuo_bets'] = {
                        'dantuo_fusion': {
                            'name': '🎯 多池融合胆拖',
                            'schemes': all_dantuo[:6],
                        }
                    }
                    print(f"[DLT-Fusion] 📊 多池融合胆拖方案生成: {len(all_dantuo)}组")
            except Exception as e:
                print(f"[DLT-Fusion] ⚠️ 胆拖方案生成跳过: {e}")

        # 输出简要摘要
        if top_n > 0 and unique:
            top = unique[0]
            print(f"[DLT-Fusion] 🎯 Top1: {top['front']} + {top['back']}  "
                  f"score={top['final_score']:.4f}  prob={top.get('hit_probability', 0):.1f}%")

        # Step 11: 📦 预测结果存档
        try:
            from prediction_store import store_prediction
            period = self._get_latest_period()
            if period:
                next_period = str(int(period) + 1)
                store_prediction(
                    next_period,
                    unique[:top_n],
                    result.get('compound_bets') if include_compound else None,
                    result.get('dan_tuo_bets'),  # 【V3.0.2】新增胆拖方案存档
                )
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 预测存档跳过: {e}")

        return result

    def predict_with_details(self, top_n: int = 5, include_compound: bool = True) -> Dict[str, Any]:
        """返回预测+详细分析"""
        pred_result = self.predict(top_n, include_compound=include_compound)
        groups = self.get_group_recommendations()

        # 构建各组详细推荐
        group_details = {}
        for gid, cands in groups.items():
            group_details[gid] = []
            for c in cands:
                gt_analysis = None
                try:
                    gt_analysis = self.game_theory.analyze_combo(c.front)
                except Exception:
                    pass
                group_details[gid].append({
                    'front': c.front,
                    'back': c.back,
                    'score': c.total_score,
                    'strategy': c.strategy_name,
                    'game_theory': gt_analysis
                })

        return {
            'single_bets': pred_result.get('single_bets', []),
            'compound_bets': pred_result.get('compound_bets', {}),
            'group_recommendations': group_details,
            'back_recommendations': self.get_back_recommendations(),
            'total_records': len(self.draws),
            'latest_draw': self.draws[-1] if self.draws else None,
            'latest_period': self._get_latest_period(),
        }

    def backtest(self, n_recent: int = 100) -> Dict[str, Any]:
        """
        重建回测：测量池级别命中率 vs 随机基准

        核心指标：
        - 每个池的平均命中个数（应该 > 随机基准 才算有效）
        - 每种复式类型的覆盖率（开奖号码在复式中的比例）
        - 对比随机基准：模型提升百分比
        """
        import random as _rnd
        _rnd.seed(42)

        if len(self.draws) < n_recent + 20:
            return {'error': f'数据不足 (需要{n_recent+20}期, 现有{len(self.draws)}期)'}

        test_draws = self.draws[-n_recent:]
        train_base = self.draws[:len(self.draws) - n_recent]

        if len(train_base) < 100:
            return {'error': '训练数据不足'}

        # 随机基准（每次选5个随机号码的前区命中期望）
        random_front_expect = 5.0 / 35 * 5  # ≈ 0.714
        random_back_expect = 2.0 / 12 * 2  # ≈ 0.333

        # 对每个测试期生成各池预测
        pool_front_hits = {'hot': [], 'cold': [], 'balance': [], 'game_theory': [], 'genetic': []}
        pool_back_hits = {'hot': [], 'cold': [], 'balance': [], 'game_theory': [], 'genetic': []}
        compound_coverage = {}

        from five_pool_sampler_complete_final import MultiPoolSampler

        for i, actual in enumerate(test_draws):
            # 用历史数据生成各池
            hist = train_base + test_draws[:i] if i > 0 else train_base
            sampler = MultiPoolSampler(hist)

            actual_front = set(actual[0])
            actual_back = set(actual[1])

            # 各池前区命中
            pool_map = {
                'hot': lambda: sampler.generate_hot_pool(20, 'front'),
                'cold': lambda: sampler.generate_cold_pool(20, 'front'),
                'balance': lambda: sampler.generate_balance_pool(20, 'front'),
                'game_theory': lambda: sampler.generate_game_theory_pool(20, 'front'),
                'genetic': lambda: sampler.generate_genetic_pool(20, 'front'),
            }
            for pname, gen in pool_map.items():
                try:
                    pool = gen()
                    pool_front_hits[pname].append(len(set(pool) & actual_front))
                except:
                    pool_front_hits[pname].append(0)

            # 各池后区命中
            pool_back_map = {
                'hot': lambda: sampler.generate_hot_pool(8, 'back'),
                'cold': lambda: sampler.generate_cold_pool(8, 'back'),
                'balance': lambda: sampler.generate_balance_pool(8, 'back'),
                'game_theory': lambda: sampler.generate_game_theory_pool(8, 'back'),
                'genetic': lambda: sampler.generate_genetic_pool(8, 'back'),
            }
            for pname, gen in pool_back_map.items():
                try:
                    pool = gen()
                    pool_back_hits[pname].append(len(set(pool) & actual_back))
                except:
                    pool_back_hits[pname].append(0)

            # 复式覆盖率
            compound_tests = [
                ('6+3', lambda: sampler.generate_6_plus_3(1, 'game_theory')),
                ('7+2', lambda: sampler.generate_7_plus_2(1, 'mixed')),
                ('8+2', lambda: sampler.generate_8_plus_2(1, 'hot')),
                ('9+3', lambda: sampler.generate_9_plus_3(1, 'cold')),
            ]
            for bt_name, bt_fn in compound_tests:
                try:
                    bets = bt_fn()
                    if bets:
                        bet = bets[0]
                        front_cov = len(set(bet['front']) & actual_front) / 5.0
                        back_cov = len(set(bet['back']) & actual_back) / 2.0
                        if bt_name not in compound_coverage:
                            compound_coverage[bt_name] = []
                        compound_coverage[bt_name].append({'front': front_cov, 'back': back_cov,
                                                        'avg': (front_cov + back_cov) / 2})
                except:
                    pass

        # 汇总结果
        result = {
            'n_tests': n_recent,
            'random_baseline': {
                'front_per_draw': round(random_front_expect, 3),
                'back_per_draw': round(random_back_expect, 3),
            },
            'pool_performance': {},
            'compound_coverage': {},
            'improvement_vs_random': {},
        }

        for pool_name in pool_front_hits:
            front_hits = pool_front_hits[pool_name]
            back_hits = pool_back_hits[pool_name]
            avg_front = round(float(np.mean(front_hits)), 3) if front_hits else 0
            avg_back = round(float(np.mean(back_hits)), 3) if back_hits else 0
            front_imp = round((avg_front / random_front_expect - 1) * 100, 1) if random_front_expect > 0 else 0
            back_imp = round((avg_back / random_back_expect - 1) * 100, 1) if random_back_expect > 0 else 0

            result['pool_performance'][pool_name] = {
                'avg_front_hits': avg_front,
                'avg_back_hits': avg_back,
                'max_front_hits': max(front_hits) if front_hits else 0,
                'max_back_hits': max(back_hits) if back_hits else 0,
            }
            result['improvement_vs_random'][pool_name] = {
                'front_improvement_%': front_imp,
                'back_improvement_%': back_imp,
            }

        for bt_name, covs in compound_coverage.items():
            front_covs = [c['front'] for c in covs]
            back_covs = [c['back'] for c in covs]
            result['compound_coverage'][bt_name] = {
                'avg_front_coverage': round(float(np.mean(front_covs)), 3),
                'avg_back_coverage': round(float(np.mean(back_covs)), 3),
                'full_hits_rate_%': round(sum(1 for c in covs if c['front'] == 1.0) / len(covs) * 100, 1) if covs else 0,
            }

        return result
    def _sample_skip_repeat_candidates(
        self, n_candidates: int = 4
    ) -> List[Dict[str, Any]]:
        """
        方案B：双期重号参考候选池
        基于draws[-1]和draws[-2]的重号模式生成候选，覆盖隔期回归号码。

        策略：
        1. 取draws[-1](上期)和draws[-2](上上期)的号码合集作为核心池
        2. 从核心池中随机选取5前2后组合
        3. 特别保留上上期号码为主体的组合（隔期回归候选）
        4. 保留上期号码为主体的组合（立即重号候选）
        5. 组合两种来源的混合候选

        Args:
            n_candidates: 生成候选数量，默认4

        Returns:
            List[Dict]: 候选列表
        """
        if len(self.draws) < 3:
            return []

        candidates = []
        prev_front = self.draws[-1][0]   # 上期前区
        prev_back = self.draws[-1][1]    # 上期后区
        skip_front = self.draws[-2][0]   # 上上期前区
        skip_back = self.draws[-2][1]    # 上上期后区

        # 核心池
        core_front = list(set(prev_front + skip_front))
        core_back = list(set(prev_back + skip_back))

        if len(core_front) < 5:
            # 补全到至少10个
            extra = [n for n in range(1, 36) if n not in core_front]
            random.shuffle(extra)
            core_front.extend(extra[:10 - len(core_front)])
        if len(core_back) < 2:
            extra = [n for n in range(1, 13) if n not in core_back]
            random.shuffle(extra)
            core_back.extend(extra[:4 - len(core_back)])

        # 生成4类候选
        strategies = []

        # 类型1：隔期回归型（以上上期为主体，混1-2个上期号码） — 对应26056→26058模式
        for _ in range(n_candidates // 2 + 1):
            base = list(skip_front)
            # 替换1个为prev_front中的随机号
            replace_i = random.randrange(len(base))
            base[replace_i] = random.choice(prev_front)
            # 后区：取上上期后区（隔期回归核心）
            back = sorted(random.sample(core_back, min(2, len(core_back))))
            strategies.append((sorted(base), back, 'skip_repeat_ago', 0.65))

        # 类型2：立即重号型（以上期为主体，混1-2个上上期号码） — 对应26057—>26058模式
        for _ in range(n_candidates // 2 + 1):
            base = list(prev_front)
            replace_i = random.randrange(len(base))
            base[replace_i] = random.choice(skip_front)
            back = sorted(random.sample(core_back, min(2, len(core_back))))
            strategies.append((sorted(base), back, 'skip_repeat_prev', 0.60))

        # 类型3：混合型（两端均匀混合）
        mixed_front = list(set(prev_front[:3] + skip_front[:3]))
        if len(mixed_front) < 5:
            extra = [n for n in range(1, 36) if n not in mixed_front]
            random.shuffle(extra)
            mixed_front.extend(extra[:5 - len(mixed_front)])
        strategies.append((sorted(mixed_front[:5]),
                           sorted(random.sample(core_back, min(2, len(core_back)))),
                           'skip_repeat_mixed', 0.62))

        for front, back, src, score in strategies:
            candidates.append({
                'front': front,
                'back': back,
                'source': src,
                'total_score': score,
                'strategy_name': f'双期参考-{src}',
            })

        if candidates:
            print(f"[DLT-Fusion] 📋 方案B：生成{len(candidates)}个双期重号参考候选")

        return candidates

    def _sample_pool_candidates(self, n_per_pool: int = 2) -> List[Dict[str, Any]]:
        """多池采样候选"""
        candidates = []
        try:
            # 使用stratified_sample生成前区组合
            front_combos = self.pool_sampler.stratified_sample(
                n_combinations=n_per_pool * 3, zone='front'
            )
            back_combos = self.pool_sampler.stratified_sample(
                n_combinations=n_per_pool * 3, zone='back'
            )

            # 交叉组合前区与后区
            for fc in front_combos[:n_per_pool * 2]:
                bc = random.choice(back_combos) if back_combos else [5, 8]
                candidates.append({
                    'front': sorted(fc),
                    'back': sorted(bc),
                    'source': 'pool_sampler',
                    'total_score': 0.5,
                    'strategy_name': 'MultiPoolSampler',
                })
        except Exception as e:
            print(f"[DLT-Fusion] 多池采样失败: {e}")

        return candidates

    def _sample_pattern_pool_candidates(self) -> List[Dict[str, Any]]:
        """模式池采样：基于跨期模式识别生成候选"""
        candidates = []
        try:
            front_pool, back_pool = self.pattern_recognizer.generate_pattern_pool(
                n_front=12, n_back=4
            )
            # 从模式池中组合多个候选
            import itertools, random
            # 从前区池中生成多组候选
            for i in range(3):
                selected_front = sorted(random.sample(front_pool, 5))
                selected_back = sorted(random.sample(back_pool, 2))
                candidates.append({
                    'front': selected_front,
                    'back': selected_back,
                    'source': 'pattern_pool',
                    'total_score': 0.6,
                    'strategy_name': '模式池-PatternPool',
                })
            # 也生成基于模式多样性的候选
            front_div, back_div = generate_pattern_diversity_pool(
                self.draws, n_front=12, n_back=4
            )
            for i in range(2):
                selected_front = sorted(random.sample(front_div, 5))
                selected_back = sorted(random.sample(back_div, 2))
                candidates.append({
                    'front': selected_front,
                    'back': selected_back,
                    'source': 'pattern_diversity',
                    'total_score': 0.55,
                    'strategy_name': '模式池-Diversity',
                })
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 模式池候选失败: {e}")
        return candidates

    def _apply_game_theory(
        self,
        all_groups: Dict[int, List[Any]],
        pool_candidates: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """博弈论优化：计算各候选的博弈论综合评分"""
        gt_scores = {}

        # 对SFE候选评分
        for gid, cands in all_groups.items():
            for c in cands:
                key = f"G{gid}_{tuple(c.front)}"
                try:
                    analysis = self.game_theory.analyze_combo(c.front)
                    # combined_score = avoidance_score + regularity_score (两者都是0-1)
                    gt_scores[key] = float(analysis['scores']['combined_score'])
                except Exception:
                    gt_scores[key] = 0.5

        # 对池采样候选评分
        for i, pc in enumerate(pool_candidates):
            key = f"P{i}_{tuple(pc['front'])}"
            try:
                analysis = self.game_theory.analyze_combo(pc['front'])
                gt_scores[key] = float(analysis['scores']['combined_score'])
            except Exception:
                gt_scores[key] = 0.5

        return gt_scores

    def _apply_genetic_optimization(
        self,
        all_groups: Dict[int, List[Any]],
        pool_candidates: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """遗传算法优化：使用进化后的适应度"""
        genetic_scores = {}

        # 获取当前最优染色体（前区+后区适应度）
        try:
            best_chromosomes = self.genetic.get_best_solutions(top_k=10)
            best_front_set = set()
            best_back_set = set()
            for chrom in best_chromosomes:
                best_front_set.update(chrom.front_numbers)
                best_back_set.update(chrom.back_numbers)
        except Exception:
            best_front_set = set(range(1, 20))  # fallback
            best_back_set = set(range(1, 8))

        # 对SFE候选评分：与最优解的重合度
        for gid, cands in all_groups.items():
            for c in cands:
                key = f"G{gid}_{tuple(c.front)}"
                try:
                    front_overlap = len(set(c.front) & best_front_set) / 5.0
                    fitness = self._chromosome_fitness(c.front, c.back)
                    genetic_scores[key] = float(fitness)
                except Exception:
                    genetic_scores[key] = 0.5

        # 对池采样候选评分
        for i, pc in enumerate(pool_candidates):
            key = f"P{i}_{tuple(pc['front'])}"
            try:
                fitness = self._chromosome_fitness(pc['front'], pc['back'])
                genetic_scores[key] = float(fitness)
            except Exception:
                genetic_scores[key] = 0.5

        return genetic_scores

    def _chromosome_fitness(self, front: List[int], back: List[int]) -> float:
        """计算组合适应度（模拟Chromosome评估）- 已集成跨期模式评分"""
        fitness = 0.0

        # 热号比例 (30%)
        hot_front = list(range(1, 20))
        hot_back = list(range(1, 8))
        front_hot = len(set(front) & set(hot_front)) / 5.0
        back_hot = len(set(back) & set(hot_back)) / 2.0
        fitness += front_hot * 0.2 + back_hot * 0.1

        # 奇偶平衡 (15%)
        front_odd = sum(1 for n in front if n % 2 == 1)
        back_odd = sum(1 for n in back if n % 2 == 1)
        front_odd_score = 1 - abs(front_odd - 3) / 3
        back_odd_score = 1 - abs(back_odd - 1.5) / 1.5
        fitness += front_odd_score * 0.10 + back_odd_score * 0.05

        # 号码分布/跨度 (10%)
        front_spread = max(front) - min(front)
        spread_score = min(front_spread / 30, 1.0)
        fitness += spread_score * 0.10

        # 连号控制 (5%)
        front_sorted = sorted(front)
        consecutive = sum(
            1 for i in range(len(front_sorted) - 1)
            if front_sorted[i+1] - front_sorted[i] == 1
        )
        consecutive_score = 1 - min(consecutive / 2, 1.0)
        fitness += consecutive_score * 0.05

        # 和值范围 (10%)
        front_sum = sum(front)
        if 90 <= front_sum <= 130:
            sum_score = 1.0
        else:
            sum_score = 1 - min(abs(front_sum - 110) / 50, 1.0)
        fitness += sum_score * 0.10

        # 跨期模式匹配度 (30%) - 新增：方案二核心
        if hasattr(self, 'pattern_recognizer') and self.pattern_recognizer._is_built:
            try:
                prev_front = self.draws[-1][0] if len(self.draws) >= 1 else None
                score_result = self.pattern_recognizer.score_combo(front, prev_front)
                pattern_fitness = score_result['total_score']
                fitness += pattern_fitness * 0.30
            except Exception:
                pass  # 模式评分失败，用其他维度补

        return max(0.0, min(1.0, fitness))

    def _collect_candidates(
        self,
        all_groups: Dict[int, List[Any]],
        pool_candidates: List[Dict[str, Any]],
        gt_scores: Dict[str, float],
        genetic_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """收集并整合所有候选"""
        all_candidates = []

        # SFE候选
        for gid, cands in all_groups.items():
            for c in cands:
                key = f"G{gid}_{tuple(c.front)}"
                all_candidates.append({
                    'front': c.front,
                    'back': c.back,
                    'base_score': getattr(c, 'total_score', 0.5),
                    'group_id': gid,
                    'strategy_name': getattr(c, 'strategy_name', 'SFE'),
                    'gt_score': gt_scores.get(key, 0.5),
                    'genetic_score': genetic_scores.get(key, 0.5),
                })

        # 池采样候选
        for i, pc in enumerate(pool_candidates):
            key = f"P{i}_{tuple(pc['front'])}"
            all_candidates.append({
                'front': pc['front'],
                'back': pc['back'],
                'base_score': pc.get('total_score', 0.5),
                'group_id': 0,
                'strategy_name': pc.get('strategy_name', 'PoolSampler'),
                'gt_score': gt_scores.get(key, 0.5),
                'genetic_score': genetic_scores.get(key, 0.5),
            })

        return all_candidates

    # ------------------------------------------------------------------
    # 【P4】📊 和值中间带补充 (Mid-Sum Compensation)
    # ------------------------------------------------------------------

    def _compensate_mid_sum(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        当候选集和值分布两极分化、中间带(100-120)覆盖率不足时，
        从平衡池/趋势池补充和值落在中间带的候选。
        """
        if not candidates or not self.draws:
            return candidates

        # 统计候选和值分布
        sums = [sum(c.get('front', [0])) for c in candidates]
        total = max(len(sums), 1)
        mid_count = sum(1 for s in sums if 100 <= s <= 120)
        mid_ratio = mid_count / total

        # 中间带占比 ≥ 20% 则不需要补偿
        if mid_ratio >= 0.20:
            return candidates

        # 需要补充多少
        target = max(int(total * 0.20) - mid_count, 2)
        print(f"[DLT-Fusion] 📊 和值中间带(100-120)覆盖率 {mid_ratio:.0%}, 需补充{target}个")

        # 从各池采样中间和值候选
        added = 0
        for _ in range(target * 3):  # 多采几次
            try:
                pool_front = self.pool_sampler.generate_balance_pool(10, 'front')
                pool_back = self.pool_sampler.generate_hot_pool(4, 'back')

                from itertools import combinations as _c
                # 从池中挑5个，检查和值
                for combo in _c(pool_front, 5):
                    s = sum(combo)
                    if 100 <= s <= 120:
                        front = sorted(combo)
                        k = tuple(front)
                        # 跳过已存在的
                        if any(tuple(c.get('front', [])) == k for c in candidates):
                            continue
                        back = sorted(pool_back[:2])
                        candidates.append({
                            'front': front,
                            'back': back,
                            'source': 'mid_sum_compensate',
                            'total_score': 0.55,
                            'base_score': 0.55,
                            'gt_score': 0.5,
                            'genetic_score': 0.5,
                            'strategy_name': '和值补偿-Balance',
                        })
                        added += 1
                        if added >= target:
                            break
            except Exception:
                pass
            if added >= target:
                break

        if added > 0:
            print(f"[DLT-Fusion] 📊 和值中间带补偿: +{added}个候选")

        return candidates

    # ------------------------------------------------------------------
    # 【优化V3.0.2】方案1补充：前区奇偶比分布补充 (Parity Enrichment)
    # ------------------------------------------------------------------

    def _enrich_parity_distribution(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        奇偶比分布补充：当候选集缺少某种奇偶比模式时，从历史数据补充

        前区奇偶比（奇:偶）历史分布（基于近1000期）：
        - 3:2 — ~35.2%（最高频）
        - 2:3 — ~28.5%
        - 4:1 — ~22.1%
        - 1:4 — ~8.3%
        - 5:0 — ~3.9%
        - 0:5 — ~2.0%

        26063期实际开奖03 15 20 29 31 奇偶比=4:1，覆盖22%的分布
        但预测5注全部为2:3或3:2，完全遗漏4:1和1:4模式

        补充策略：
        - 统计候选目前的奇偶比分布
        - 检测缺少哪些模式（特别是4:1和1:4）
        - 从历史匹配候选或随机生成补充
        """
        if not candidates:
            return candidates

        # 统计当前候选的奇偶比分布
        parity_counts = Counter()
        for c in candidates:
            front = c.get('front', [])
            if len(front) == 5:
                odd = sum(1 for n in front if n % 2 == 1)
                even = 5 - odd
                parity_counts[f"{odd}:{even}"] += 1

        total = max(len(candidates), 1)

        # 目标最低覆盖率（每种模式至少有不少于此比例的候选）
        # 4:1 和 1:4 经常被忽略，需要强制补充
        TARGET_RATIOS = {
            '3:2': 0.25,  # 至少25%
            '2:3': 0.15,  # 至少15%
            '4:1': 0.10,  # 至少10% ← 26063缺失
            '1:4': 0.05,  # 至少5%  ← 常见缺失
            '5:0': 0.02,  # 至少2%  （极低概率但保留）
            '0:5': 0.02,  # 至少2%
        }

        # 检查哪些模式需要补充
        to_add = []
        for ratio, target_pct in TARGET_RATIOS.items():
            current_pct = parity_counts.get(ratio, 0) / total
            if current_pct < target_pct:
                target_count = max(int(total * target_pct) - parity_counts.get(ratio, 0), 1)
                to_add.append((ratio, target_count))

        if not to_add:
            return candidates

        print(f"[DLT-Fusion] 🎯 奇偶比分布补充: {to_add}")

        # 为每个缺失模式生成候选
        new_candidates = []
        ZONE1 = set(range(1, 13))
        ZONE2 = set(range(13, 25))
        ZONE3 = set(range(25, 36))

        for ratio_str, need in to_add:
            odd_target = int(ratio_str.split(':')[0])
            even_target = 5 - odd_target

            # 用历史数据中最接近的号码生成
            for _ in range(need * 3):  # 多生成一些以备去重
                # 从不向池中按奇偶筛选
                all_odds = [n for n in range(1, 36) if n % 2 == 1]
                all_evens = [n for n in range(1, 36) if n % 2 == 0]

                if odd_target > len(all_odds) or even_target > len(all_evens):
                    continue

                # 随机选odd_target个奇数和even_target个偶数
                chosen_odds = random.sample(all_odds, odd_target)
                chosen_evens = random.sample(all_evens, even_target)

                front_candidate = sorted(chosen_odds + chosen_evens)

                # 检查三区覆盖（避免过度聚集在某区）
                front_set = set(front_candidate)
                z1c = 1 if front_set & ZONE1 else 0
                z2c = 1 if front_set & ZONE2 else 0
                z3c = 1 if front_set & ZONE3 else 0
                zone_cov = z1c + z2c + z3c

                # 至少覆盖2个区
                if zone_cov < 2:
                    # 尝试替换一个号码到缺失区
                    for z_pool in [ZONE1, ZONE2, ZONE3]:
                        missing_in_zone = z_pool - front_set
                        has_in_zone = front_set & z_pool
                        if not has_in_zone and missing_in_zone:
                            # 替换一个与缺失区间号同奇偶的号码
                            to_replace_idx = random.randint(0, 4)
                            old_val = front_candidate[to_replace_idx]
                            candidates_in_zone = [n for n in missing_in_zone if n % 2 == old_val % 2]
                            if candidates_in_zone:
                                new_val = random.choice(candidates_in_zone)
                                front_candidate[to_replace_idx] = new_val
                                front_candidate = sorted(front_candidate)
                            break

                # 检查是否已存在
                key = tuple(front_candidate)
                if key not in set(tuple(c.get('front', [])) for c in candidates + new_candidates):
                    new_candidates.append({
                        'front': front_candidate,
                        'back': [1, 12],  # placeholder, 后续分配
                        'base_score': 0.55,
                        'group_id': 0,
                        'strategy_name': f'奇偶补充-{ratio_str}',
                        'gt_score': 0.5,
                        'genetic_score': 0.5,
                    })
                    if len(new_candidates) >= need:
                        break

        if new_candidates:
            print(f"[DLT-Fusion] 🎯 奇偶比补充: 新增{len(new_candidates)}个候选 "
                  f"({', '.join(r for r, _ in to_add)})")

        return candidates + new_candidates

    def _compute_final_scores(self, candidates: List[Dict[str, Any]]) -> None:
        """计算综合评分（后续会被neural评分叠加修改）"""
        for c in candidates:
            # 综合评分: base*0.4 + gt*0.3 + genetic*0.3
            c['final_score'] = c['base_score'] * 0.4 + c['gt_score'] * 0.3 + c['genetic_score'] * 0.3

    def _deduplicate_and_assign_back(
        self,
        candidates: List[Dict[str, Any]],
        back_recs: List[List[int]]
    ) -> List[Dict[str, Any]]:
        """去重并分配后区（P3: 增加后区覆盖扩宽）"""
        seen = set()
        unique = []

        # P3: 后区覆盖扩宽 — 确保Top5的后区多样性
        # 将候选按分数排序，优先给高分分配不同后区
        scored = [(c, c.get('final_score', c.get('base_score', 0.5)))
                   for c in candidates]
        scored.sort(key=lambda x: -x[1])

        if not back_recs:
            back_recs = [[1, 12]]

        back_index = 0
        used_back_pairs = []       # 已分配的后区对（去重用）
        wide_start = 0             # 何时开始扩宽模式
        back_strategy = 'normal'   # 'normal' → 'wide' → 'fill'

        for c, _ in scored:
            k = tuple(c['front'])
            if k in seen:
                continue
            seen.add(k)

            # P3: 分配策略 — 前几个用高分后区，中间扩宽，后续填充
            if len(unique) < len(back_recs):
                # normal: 按顺序分配
                idx = back_index % len(back_recs)
                c['back'] = back_recs[idx]
                back_index += 1

            # wide: 前8个之后，尝试分配与已有后区不同的组合
            # 从候选后区中选一个此前没出现过的，如果都出现过则用当前
            if len(unique) >= wide_start or back_strategy == 'wide':
                if back_strategy == 'normal' and len(unique) >= 5:
                    # 前5个用标准分配后，进入扩宽模式
                    back_strategy = 'wide'
                    wide_start = len(unique)

            if back_strategy == 'wide' and len(back_recs) > 1:
                # 找还没用过的后区对
                unused = [b for b in back_recs
                          if tuple(b) not in used_back_pairs]
                if unused:
                    c['back'] = unused[len(unique) % len(unused)]
                else:
                    c['back'] = back_recs[len(unique) % len(back_recs)]
            elif back_strategy == 'normal':
                c['back'] = back_recs[back_index % len(back_recs)]
                back_index += 1

            used_back_pairs.append(tuple(c['back']))

            # 至少确保前6个覆盖不同后区对
            if len(unique) < 6 and len(used_back_pairs) >= 2:
                # 检查当前后区是否重复了
                current = tuple(c['back'])
                count_before = used_back_pairs[:-1].count(current)
                if count_before >= 2:
                    # 已出现2次，换一个
                    alt = [b for b in back_recs if tuple(b) != current]
                    if alt:
                        c['back'] = alt[len(unique) % len(alt)]
                        used_back_pairs[-1] = tuple(c['back'])

            unique.append(c)

        return unique

    # ------------------------------------------------------------------
    # 【方案A】隔期重号评分增强 (Skip-Repeat Boost)
    # ------------------------------------------------------------------

    def _apply_skip_repeat_boost(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        隔期重号评分增强：候选与draws[-2](上上期)号码匹配时加分。

        历史统计：隔期前区重号≥2个的概率约14.6%，且经常伴随后区隔期回归。

        # 【优化V3.0.2】提高隔期重号加分倍率，因为26063实际重号29和31说明隔期重复是有效信号
        增强策略：
        - 前区与上上期匹配≥2个：最终评分×1.20
        - 后区与上上期匹配≥1个：最终评分×1.15
        - 前后区同时匹配（前≥2且后≥1）：最终评分×1.28
        - 前区匹配≥3个：最终评分×1.18

        Returns:
            评分调整后的候选列表
        """
        if len(self.draws) < 3:
            return candidates

        skip_front = set(self.draws[-2][0])  # 上上期前区
        skip_back = set(self.draws[-2][1])   # 上上期后区
        boost_count = 0
        strong_boost_count = 0

        for c in candidates:
            c_front = set(c.get('front', []))
            c_back = set(c.get('back', []))

            front_skip_overlap = len(c_front & skip_front)
            back_skip_overlap = len(c_back & skip_back)

            orig = c.get('final_score', c.get('base_score', 1.0))
            multiplier = 1.0
            reasons = []

            if front_skip_overlap >= 2 and back_skip_overlap >= 1:
                # 前后区同时隔期匹配：强增强
                multiplier = 1.28
                reasons.append(f'前后隔期双匹配(前{front_skip_overlap}+后{back_skip_overlap})')
                strong_boost_count += 1
            elif front_skip_overlap >= 3:
                # 前区高度匹配上上期：中等增强
                multiplier = 1.18
                reasons.append(f'前区隔期高匹配({front_skip_overlap}个)')
            elif front_skip_overlap >= 2:
                # 前区匹配≥2：基础增强
                multiplier = 1.20
                reasons.append(f'前区隔期匹配({front_skip_overlap}个)')
            elif back_skip_overlap >= 1:
                # 后区单独匹配≥1：轻度增强
                multiplier = 1.15
                reasons.append(f'后区隔期匹配({back_skip_overlap}个)')

            if multiplier > 1.0:
                c['final_score'] = orig * multiplier
                boost_count += 1
                if boost_count <= 5:
                    print(f"[DLT-Fusion] 🚀 隔期重号增强: 前区{c.get('front', [])} "
                          f"({', '.join(reasons)}, score: {orig:.4f}→{c['final_score']:.4f}, ×{multiplier:.2f})")

        if boost_count > 0:
            print(f"[DLT-Fusion] 🚀 隔期重号增强合计: {boost_count}注受加成 "
                  f"(强增强{strong_boost_count}注)")

        return candidates

    # ------------------------------------------------------------------
    # 【方案D】智能重号惩罚 (Smart Repeat Penalty)
    # ------------------------------------------------------------------

    def _apply_repeat_penalty(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        【优化 V2.1】智能重号惩罚（动态判断版）

        删除"期望重号1个"的固定预期，改为基于上期和值的动态判断：
        - 上期和值偏高(>120) → 倾向于无重号（大和值后号码切换概率高）
        - 上期和值偏低(<70) → 倾向于0-1个重号
        - 上期和值中等(70-120) → 正常处理

        区分预期重号与热号堆叠判定逻辑：
        - 重叠数≥3且其中热号重号占多数（≥2个热号）→ 5%折扣
        - 重叠数≥3但以冷号为主（≤1个热号）→ 视为正常趋势，不打折

        冷热度判定：号码在最近10期出现≥4次 = 热号
        """
        if len(self.draws) < 2:
            return candidates

        latest_front = set(self.draws[-1][0])
        latest_sum = sum(self.draws[-1][0])

        # ---- 动态重号预期 ----
        # 上期和值高(>120) → 强烈预期无重号，降低所有重号容忍度
        # 上期和值低(<70) → 温和预期少重号
        # 上期和值中 → 正常
        if latest_sum > 120:
            repeat_expected = 0        # 无重号预期
            repeat_tolerance = 1        # 最多容忍1个重号
        elif latest_sum < 70:
            repeat_expected = 0
            repeat_tolerance = 2
        else:
            repeat_expected = 1
            repeat_tolerance = 2

        # 计算最近10期每个号码的出现频率
        recent_window = 10
        recent_front_all = []
        for i in range(max(0, len(self.draws) - recent_window), len(self.draws)):
            recent_front_all.extend(self.draws[i][0])
        num_freq = Counter(recent_front_all)
        HOT_FREQ_THRESHOLD = 4

        penalty_count = 0
        neutral_count = 0

        for c in candidates:
            c_front = set(c.get('front', []))
            overlap_nums = c_front & latest_front
            overlap = len(overlap_nums)

            if overlap < 3:
                continue

            # 分析每个重叠号码的"热程度"
            hot_overlaps = sum(1 for n in overlap_nums if num_freq.get(n, 0) >= HOT_FREQ_THRESHOLD)
            cold_overlaps = overlap - hot_overlaps

            orig = c.get('final_score', c.get('base_score', 0.5))

            # 【优化V3.0.2】降低热号堆叠惩罚强度，减少对合理重号的误伤
            # 26063实际开奖含重号29和31（若来自26062），说明3个个重号是可能的
            penalty_mult = 1.0
            reasons = []

            if hot_overlaps >= 3:
                # 强热号堆叠（≥3个热号）：仅当时期和值>120时惩罚，否则宽松处理
                if repeat_expected == 0:
                    penalty_mult = 0.93  # 从0.90放松到0.93
                    reasons.append(f'高和值热号堆叠×0.93')
                else:
                    penalty_mult = 0.97  # 从0.95放松到0.97
                    reasons.append(f'热号堆叠×0.97')
            elif hot_overlaps == 2:
                # 2个热号堆叠：仅轻微降分（原为惩罚5%，现仅2%）
                if repeat_expected == 0:
                    penalty_mult = 0.97
                    reasons.append(f'高和值双热号×0.97')
                else:
                    # 2个合理重号可以接受，不惩罚
                    penalty_mult = 1.0

            # 冷号为主的重号：趋势延续加分（保持，略微提高）
            if cold_overlaps >= 2 and overlap <= repeat_tolerance + 1:
                if hot_overlaps < 2:
                    penalty_mult *= 1.04  # 从1.03提高到1.04
                    reasons.append(f'冷号趋势延续×1.04')

            # 1个热号+1个冷号混合重号：中性（不惩罚，小加）
            if overlap == 2 and hot_overlaps == 1 and cold_overlaps == 1:
                if penalty_mult == 1.0:
                    penalty_mult = 1.02
                    reasons.append(f'混合重号×1.02')

            if penalty_mult != 1.0:
                c['final_score'] = orig * penalty_mult
                penalty_count += 1
                if penalty_count <= 3:
                    print(f"[DLT-Fusion] 🔽 重号调整: 前区{c.get('front', [])} "
                          f"重叠{overlap}个(热{hot_overlaps}/冷{cold_overlaps}, "
                          f"上期和值={latest_sum}) "
                          f"({' '.join(reasons)}, score: {orig:.4f}→{c['final_score']:.4f})")

        if penalty_count > 0:
            print(f"[DLT-Fusion] 🔽 重号惩罚(方案D-动态)合计: {penalty_count}注受折扣 "
                  f"(上期和值={latest_sum}, 预期重号={repeat_expected})")
        if neutral_count > 0:
            print(f"[DLT-Fusion] ✅ 趋势延续识别合计: {neutral_count}注受加成")

        return candidates

    def _filter_recent_draws(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        过滤掉与近期开奖号码高度重复的候选组合。
        核心逻辑：
        1. 前区5个号码与最新一期完全一致 → 直接剔除（连续两期前区相同的概率极低）
        2. 前区与最新一期重合≥4个 → 降权而非剔除（保留可能性但不作为首选）
        3. 后区与最新一期完全一致 → 降权
        """
        if len(self.draws) < 2:
            return candidates

        latest_front = set(self.draws[-1][0])
        latest_back = set(self.draws[-1][1])

        # 同时也检查倒数第2期
        prev_front = set(self.draws[-2][0]) if len(self.draws) >= 2 else set()

        filtered = []
        for c in candidates:
            c_front = set(c.get('front', []))
            c_back = set(c.get('back', []))

            # 规则1：前区与最新一期完全相同 → 剔除
            if c_front == latest_front:
                print(f"[DLT-Fusion] ⛔ 过滤: 前区{c['front']}与上期完全相同")
                continue

            # 规则2：前区与最新一期重合≥4个 → 大幅降权
            front_overlap = len(c_front & latest_front)
            if front_overlap >= 4:
                orig_score = c.get('final_score', c.get('base_score', 0.5))
                c['final_score'] = orig_score * 0.3
                c['base_score'] = c.get('base_score', 0.5) * 0.3
                print(f"[DLT-Fusion] ⚠️ 降权: 前区{c['front']}与上期重合{front_overlap}个 (score: {orig_score:.4f}→{c['final_score']:.4f})")

            # 规则3：前区与倒数第2期完全相同 → 剔除
            if prev_front and c_front == prev_front:
                print(f"[DLT-Fusion] ⛔ 过滤: 前区{c['front']}与倒数第2期完全相同")
                continue

            # 规则4：后区与最新一期完全相同 → 降权
            if c_back == latest_back:
                orig_score = c.get('final_score', c.get('base_score', 0.5))
                c['final_score'] = orig_score * 0.5
                c['base_score'] = c.get('base_score', 0.5) * 0.5
                print(f"[DLT-Fusion] ⚠️ 降权: 后区{c['back']}与上期完全相同 (score: {orig_score:.4f}→{c['final_score']:.4f})")

            filtered.append(c)

        return filtered

    def _get_latest_period(self) -> Optional[str]:
        """获取最近一期期号（直接从Excel读取）"""
        try:
            import pandas as pd
            df = pd.read_excel(self.data_path)
            if '期号' in df.columns:
                # load_dlt_data 检测到倒序时会反转，这里直接取最后一行（最新）
                first = int(df.iloc[0]['期号'])
                last = int(df.iloc[-1]['期号'])
                latest = str(max(first, last))
                return latest
            return None
        except Exception:
            return None

    # ==================================================================
    # 【优化 V2.2.0】方案1：区间分布均衡评分 (Zone Balance Scoring)
    # 针对26061期分析结论：实际开奖01 10 12 26 35覆盖3个区间，
    # 模型前区分布偏离中区，小号(01-12)和大号(30-35)覆盖不足。
    # 新增区间覆盖率与区间散度评分，奖励跨区间分布的候选。
    # ==================================================================

    def _apply_zone_balance_scoring(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        区间分布均衡评分：奖励跨区间分布的候选，惩罚集中在单一区间的候选。

        三区划分：Z1=01-12, Z2=13-24, Z3=25-35

        核心逻辑：
        - 计算每个候选覆盖的区间数 (zone_coverage): 1~3
        - 计算每个候选在各区间的号码分布均匀度 (zone_entropy)
        - 历史胜率统计显示：3区覆盖的命中率 > 2区 > 1区
        - 此外增加小号区(01-12)和超大号区(30-35)的专项覆盖检查
        - 调整幅度：±2%~6% 基于覆盖模式
        """
        ZONE1 = set(range(1, 13))
        ZONE2 = set(range(13, 25))
        ZONE3 = set(range(25, 36))

        # 统计最近50期的三区覆盖模式分布
        window = min(50, len(self.draws))
        hist_zone_coverage = []
        hist_small_count = []          # 01-12号码个数
        hist_large_count = []          # 30-35号码个数
        for i in range(max(0, len(self.draws) - window), len(self.draws)):
            s = set(self.draws[i][0])
            zc = (1 if s & ZONE1 else 0) + (1 if s & ZONE2 else 0) + (1 if s & ZONE3 else 0)
            hist_zone_coverage.append(zc)
            hist_small_count.append(len(s & ZONE1))
            hist_large_count.append(len(s & {30, 31, 32, 33, 34, 35}))

        avg_zone_coverage = np.mean(hist_zone_coverage) if hist_zone_coverage else 2.2
        avg_small_count = np.mean(hist_small_count) if hist_small_count else 1.2
        avg_large_count = np.mean(hist_large_count) if hist_large_count else 0.6

        boost_count = 0
        penalty_count = 0

        for c in candidates:
            front = set(c.get('front', []))
            if not front or len(front) < 5:
                continue

            z1c = len(front & ZONE1)
            z2c = len(front & ZONE2)
            z3c = len(front & ZONE3)

            # 区间覆盖率：覆盖了3个区间
            zone_coverage = (1 if z1c > 0 else 0) + (1 if z2c > 0 else 0) + (1 if z3c > 0 else 0)

            # 小号区(01-12)实际个数 vs 历史均值
            small_deviation = z1c - avg_small_count
            # 大号区(30-35)实际个数 vs 历史均值
            large_deviation = len(front & {30, 31, 32, 33, 34, 35}) - avg_large_count

            orig = c.get('final_score', c.get('base_score', 0.5))
            adjustment = 0.0
            reasons = []

            # 1. 区间覆盖率评分（主力信号）
            if zone_coverage == 3:
                # 全覆盖3个区间 → 奖励3%（最符合实际开奖规律）
                adjustment += 0.03
                reasons.append('3区全覆盖')
            elif zone_coverage == 2:
                # 覆盖2个区间 → 中性偏正1%
                adjustment += 0.01
                reasons.append('2区覆盖')
            else:
                # 仅1个区间 → 惩罚3%
                adjustment -= 0.03
                reasons.append('仅1区')

            # 2. 分布均匀度评分：号码过于集中在某个区间则减分
            zone_counts = sorted([z1c, z2c, z3c], reverse=True)
            if len(zone_counts) >= 2 and zone_counts[0] >= 4:
                # 某个区间有≥4个号码 → 非常集中，惩罚2%
                adjustment -= 0.02
                reasons.append(f'{zone_counts[0]}号聚一区')

            # 3. 小号区覆盖检查（针对26061号组01 10 12 26 35）
            if small_deviation <= -1:
                # 小号个数低于历史均值1个以上 → 加点小号覆盖
                if z1c == 0:
                    adjustment -= 0.01
                    reasons.append('缺小号')

            # 4. 超大号(30-35)覆盖检查
            if large_deviation <= -0.5 and z3c < 1:
                # 超大号覆盖不足
                adjustment -= 0.005
                reasons.append('缺大尾')

            # 5. 边界：调整幅度限制在±6%
            adjustment = max(-0.06, min(0.06, adjustment))

            c['final_score'] = max(0.1, orig * (1.0 + adjustment))
            c['zone_balance_adj'] = adjustment

            if adjustment > 0.005:
                boost_count += 1
            elif adjustment < -0.005:
                penalty_count += 1

        if boost_count > 0 or penalty_count > 0:
            print(f"[DLT-Fusion] 📐 区间均衡评分: +{boost_count}注加分 / {penalty_count}注减分 "
                  f"(历史均值: {avg_zone_coverage:.1f}区覆盖, "
                  f"小号{avg_small_count:.1f}个, 大尾{avg_large_count:.1f}个)")

        return candidates

    # ==================================================================
    # 【优化 V2.2.0】方案4：前区散度特征评分 (Scatter Scoring)
    # 针对26061期分析结论：实际开奖01 10 12 26 35呈现高散度/
    # 无规律散乱分布特征。新增散度指标评估候选号码的"散乱程度"，
    # 匹配历史散乱型开奖的模式。
    # ==================================================================

    def _apply_scatter_scoring(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        前区散度特征评分：评估号码组合的散乱/聚集程度。

        散度指标：
        1. 最大间隔 (max_gap)：相邻排序号之间的最大差值
           (01,10,12,26,35 → gaps: 9,2,14,9 → max_gap = 14)
        2. 间隔方差 (gap_variance)：间隔值的方差，衡量分布不均匀度
        3. 最小间隔 (min_gap)：相邻最小差值
        4. 区间散度 (zone_entropy)：号码在三区间分布的香农熵
        5. 平均间隔 (avg_gap)：所有相邻间隔的平均值

        逻辑：
        - 统计历史散度指标分布
        - 候选的散度指标越接近历史典型散乱分布 → 加分
        - 极端值（所有号码扎堆或极度离散）→ 减分
        """
        window = min(100, len(self.draws))
        if len(self.draws) < 10:
            return candidates

        # 统计历史散度分布
        hist_max_gaps, hist_gap_variances, hist_min_gaps, hist_avg_gaps = [], [], [], []
        for i in range(max(0, len(self.draws) - window), len(self.draws)):
            s = sorted(self.draws[i][0])
            gaps = [s[j+1] - s[j] for j in range(len(s) - 1)]
            if gaps:
                hist_max_gaps.append(max(gaps))
                hist_gap_variances.append(np.var(gaps))
                hist_min_gaps.append(min(gaps))
                hist_avg_gaps.append(np.mean(gaps))

        # 计算历史散度指标的分位数作为评分基准
        def get_percentile_score(value: float, hist: List[float]) -> float:
            """返回0-1分：值在历史分布中的合理程度（中间80%区域得分高）"""
            if len(hist) < 20:
                return 0.5
            p10 = np.percentile(hist, 10)
            p25 = np.percentile(hist, 25)
            p75 = np.percentile(hist, 75)
            p90 = np.percentile(hist, 90)

            if p10 <= value <= p90:
                return 1.0  # 主流区间 → 满分
            elif p25 <= value <= p75:
                return 0.8  # 核心区间 → 良好
            elif value < p10 or value > p90:
                return 0.3  # 极端值 → 低分
            return 0.5

        # 典型散乱特征的模式模板（基于26061分析）
        # 散乱型开奖特征：max_gap >= 10, gap_variance >= 25, min_gap <= 2
        def is_scattered_pattern(max_g: float, var_g: float, min_g: float) -> bool:
            """判断是否符合散乱型模式"""
            return max_g >= 10 and var_g >= 20 and min_g <= 2

        boost_count = 0
        penalty_count = 0

        for c in candidates:
            front = sorted(c.get('front', []))
            if not front or len(front) < 5:
                continue

            gaps = [front[j+1] - front[j] for j in range(len(front) - 1)]
            if not gaps:
                continue

            max_gap = max(gaps)
            min_gap = min(gaps)
            gap_variance = float(np.var(gaps))
            avg_gap = float(np.mean(gaps))

            orig = c.get('final_score', c.get('base_score', 0.5))
            adjustment = 0.0
            reasons = []

            # 1. 最大间隔评分
            max_gap_score = get_percentile_score(max_gap, hist_max_gaps)
            if max_gap_score >= 0.8:
                adjustment += 0.01
            elif max_gap_score <= 0.3:
                adjustment -= 0.01
                reasons.append('最大间隔异常')

            # 2. 间隔方差评分
            var_score = get_percentile_score(gap_variance, hist_gap_variances)
            if var_score >= 0.8:
                adjustment += 0.01
            elif var_score <= 0.3:
                adjustment -= 0.015
                reasons.append('间隔方差极端')

            # 3. 识别散乱型模式（针对26061开奖特征）
            if is_scattered_pattern(max_gap, gap_variance, min_gap):
                # 散乱型 → 加分2%（近期散乱开奖增多趋势）
                adjustment += 0.02
                reasons.append('散乱分布')
            elif max_gap <= 6 and gap_variance <= 10:
                # 过于聚集 → 减分2%
                adjustment -= 0.02
                reasons.append('过度聚集')

            # 4. 最小间隔检查：全为连号的情况惩罚
            consecutive_count = sum(1 for g in gaps if g == 1)
            if consecutive_count >= 3:
                adjustment -= 0.02
                reasons.append(f'{consecutive_count}组连号')

            # 5. 最大间隔的位置检查：若最大间隔在后半区(已过中位数) → 更符合散乱型
            # 26061的最大间隔在12→26，跨越了中区
            max_gap_idx = gaps.index(max_gap)
            if max_gap_idx >= 2 and max_gap >= 10:
                adjustment += 0.005
                reasons.append('中后段大裂口')

            # 边界控制
            adjustment = max(-0.06, min(0.06, adjustment))

            c['final_score'] = max(0.1, orig * (1.0 + adjustment))
            c['scatter_adj'] = adjustment

            if adjustment > 0.005:
                boost_count += 1
            elif adjustment < -0.005:
                penalty_count += 1

        if boost_count > 0 or penalty_count > 0:
            print(f"[DLT-Fusion] 📊 散度特征评分: +{boost_count}注加分 / {penalty_count}注减分 "
                  f"(历史: 最大间隔中位={np.median(hist_max_gaps):.0f}, "
                  f"方差中位={np.median(hist_gap_variances):.1f}, "
                  f"n={len(hist_max_gaps)})")

        return candidates

    # ==================================================================
    # 【优化 V2.1.0】特征工程补充：尾号聚合检测 + AC值跟踪
    # ==================================================================

    def _apply_tail_density_scoring(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        尾号聚合检测：评估候选号码的尾号分布密集度。

        逻辑：
        - 计算候选前区的尾号集合（号码个位数）
        - 正常分布应有4-5个不同尾号（含1个重复尾号）
        - 尾号分布过于集中（≤3个不同尾号）→ 降分
        - 有1个重复尾号(4个不同尾号) → 正常，小幅加分

        评分调整幅度：±3%
        """
        for c in candidates:
            front = c.get('front', [])
            if not front or len(front) < 5:
                continue

            tails = [n % 10 for n in front]
            unique_tails = len(set(tails))

            orig = c.get('final_score', c.get('base_score', 0.5))

            if unique_tails <= 3:
                c['final_score'] = orig * 0.97
                c['tail_density_adj'] = -0.03
            elif unique_tails == 5:
                pass
            elif unique_tails == 4:
                c['final_score'] = orig * 1.01
                c['tail_density_adj'] = 0.01

        return candidates

    def _apply_ac_value_scoring(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        AC值跟踪评分：基于历史AC值分布调整候选评分。

        逻辑：
        - 统计最近100期的AC值分布
        - 计算AC值的概率密度
        - AC值在历史高频区间(通常4-8)的候选加分
        - AC值在低频区间的候选减分
        """
        # 统计最近100期AC值分布
        window = min(100, len(self.draws))
        ac_values = []
        for i in range(max(0, len(self.draws) - window), len(self.draws)):
            front = self.draws[i][0]
            diffs = set()
            for j in range(len(front)):
                for k in range(j + 1, len(front)):
                    diffs.add(abs(front[k] - front[j]))
            ac = len(diffs) - (len(front) - 1)
            ac_values.append(ac)

        from collections import Counter as _Counter
        ac_dist = _Counter(ac_values)
        total = sum(ac_dist.values()) or 1
        ac_prob = {k: v / total for k, v in ac_dist.items()}
        median_ac = sorted(ac_values)[len(ac_values) // 2] if ac_values else 6

        for c in candidates:
            front = c.get('front', [])
            if not front or len(front) < 5:
                continue

            # 计算该候选的AC值
            s = sorted(front)
            diffs = set()
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    diffs.add(abs(s[j] - s[i]))
            ac = len(diffs) - (len(s) - 1)

            orig = c.get('final_score', c.get('base_score', 0.5))
            prob = ac_prob.get(ac, 0.05)

            # 调整幅度：基于概率的线性调整
            if prob >= 0.15:
                # 高频AC值 → 加1%
                c['final_score'] = orig * 1.01
                c['ac_adj'] = 0.01
            elif prob <= 0.03:
                # 低频AC值 → 减2%
                c['final_score'] = orig * 0.98
                c['ac_adj'] = -0.02
            # 中等频率 → 不做调整

        return candidates

    # ------------------------------------------------------------------
    # 【P1】🎯 号码过度集中抑制 (Diversity Penalty)
    # ------------------------------------------------------------------

    def _apply_diversity_penalty(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        抑制号码过度集中在少数热号上。

        统计候选集中每个号码出现的频次，如果某号码出现比例超过阈值，
        则在包含该号码的候选中降低其 final_score。

        阈值逻辑：
        - 出现比例 > 40% 且 < 60%: final_score × 0.93
        - 出现比例 ≥ 60%: final_score × 0.85
        """
        from collections import Counter
        total = max(len(candidates), 1)

        # 统计号码频率
        num_counter = Counter()
        for c in candidates:
            for n in c.get('front', []):
                num_counter[n] += 1

        penalty_count = 0
        heavy_penalty_count = 0

        for c in candidates:
            front = c.get('front', [])
            # 取该候选受罚最重的号码比例
            max_ratio = max((num_counter.get(n, 0) / total) for n in front) if front else 0

            if max_ratio >= 0.60:
                orig = c.get('final_score', c.get('base_score', 0.5))
                c['final_score'] = orig * 0.85
                c['diversity_penalty'] = -0.15
                heavy_penalty_count += 1
                penalty_count += 1
            elif max_ratio >= 0.40:
                orig = c.get('final_score', c.get('base_score', 0.5))
                c['final_score'] = orig * 0.93
                c['diversity_penalty'] = -0.07
                penalty_count += 1

        if penalty_count > 0:
            print(f"[DLT-Fusion] 🎯 多样性惩罚: {penalty_count}注受罚 "
                  f"(重度{heavy_penalty_count}注, 阈值40%/60%)")

        return candidates

    # ==================================================================
    # 【优化 V2.1.0】融合模型输出校准：偏差仪表盘 + 置信度重校准
    # ==================================================================

    def _compute_deviation_dashboard(self, candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        偏差仪表盘：计算本轮候选集相对于历史数据的偏差指标。

        四个核心指标：
        1. 和值偏差(sum_deviation) — 候选平均和值 vs 最近20期平均和值
        2. 跨度偏差(span_deviation) — 候选平均跨度 vs 最近20期平均跨度
        3. 重号偏差(repeat_deviation) — 候选平均重号数 vs 最近20期平均重号数
        4. 大小号偏差(size_deviation) — 候选平均大号占比 vs 最近20期平均大号占比
        """
        window = min(20, len(self.draws))
        if len(self.draws) < window:
            return {'sum_deviation': 0, 'span_deviation': 0,
                    'repeat_deviation': 0, 'size_deviation': 0}

        # 历史数据统计
        recent = self.draws[-window:]
        hist_sums = [sum(d[0]) for d in recent]
        hist_spans = [max(d[0]) - min(d[0]) for d in recent]
        hist_repeats = []
        for i in range(1, len(recent)):
            overlap = len(set(recent[i][0]) & set(recent[i - 1][0]))
            hist_repeats.append(overlap)
        hist_big = [sum(1 for n in d[0] if n >= 25) / 5.0 for d in recent]
        # P5: 记录三区分布（1-12区占比 / 25-35区占比）
        hist_zone1 = [sum(1 for n in d[0] if n <= 12) / 5.0 for d in recent]
        hist_zone3 = [sum(1 for n in d[0] if n >= 25) / 5.0 for d in recent]

        avg_hist_sum = float(np.mean(hist_sums))
        avg_hist_span = float(np.mean(hist_spans))
        avg_hist_repeat = float(np.mean(hist_repeats)) if hist_repeats else 1
        avg_hist_big = float(np.mean(hist_big))

        # 候选数据统计
        if not candidates:
            return {'sum_deviation': 0, 'span_deviation': 0,
                    'repeat_deviation': 0, 'size_deviation': 0}

        cand_sums = [sum(c.get('front', [])) for c in candidates]
        cand_spans = []
        for c in candidates:
            f = c.get('front', [])
            cand_spans.append(max(f) - min(f) if f else 0)
        cand_big = []
        cand_zone1 = []
        cand_zone3 = []
        for c in candidates:
            f = c.get('front', [])
            cand_big.append(sum(1 for n in f if n >= 25) / 5.0 if f else 0)
            cand_zone1.append(sum(1 for n in f if n <= 12) / 5.0 if f else 0)
            cand_zone3.append(sum(1 for n in f if n >= 25) / 5.0 if f else 0)

        avg_cand_sum = float(np.mean(cand_sums)) if cand_sums else 0
        avg_cand_span = float(np.mean(cand_spans)) if cand_spans else 0
        avg_cand_big = float(np.mean(cand_big)) if cand_big else 0

        # 计算偏差值（标准化为-1到1）
        sum_dev = (avg_cand_sum - avg_hist_sum) / max(avg_hist_sum, 1)
        span_dev = (avg_cand_span - avg_hist_span) / max(avg_hist_span, 1)
        repeat_dev = 0  # 无法直接计算候选重号数
        size_dev = (avg_cand_big - avg_hist_big) / max(avg_hist_big, 0.1)
        # P5: 三区偏差（1-12区 = 小号, 25-35区 = 大号）
        avg_hist_zone1 = float(np.mean(hist_zone1)) if hist_zone1 else 0.33
        avg_hist_zone3 = float(np.mean(hist_zone3)) if hist_zone3 else 0.33
        avg_cand_zone1 = float(np.mean(cand_zone1)) if cand_zone1 else 0
        avg_cand_zone3 = float(np.mean(cand_zone3)) if cand_zone3 else 0
        zone1_dev = (avg_cand_zone1 - avg_hist_zone1) / max(avg_hist_zone1, 0.1)
        zone3_dev = (avg_cand_zone3 - avg_hist_zone3) / max(avg_hist_zone3, 0.1)

        # 限制范围
        sum_dev = max(-1.0, min(1.0, sum_dev))
        span_dev = max(-1.0, min(1.0, span_dev))
        size_dev = max(-1.0, min(1.0, size_dev))
        zone1_dev = max(-1.0, min(1.0, zone1_dev))
        zone3_dev = max(-1.0, min(1.0, zone3_dev))

        dashboard = {
            'sum_deviation': round(sum_dev, 4),
            'span_deviation': round(span_dev, 4),
            'repeat_deviation': round(repeat_dev, 4),
            'size_deviation': round(size_dev, 4),
            # P5: 三区偏差
            'zone1_deviation': round(zone1_dev, 4),
            'zone3_deviation': round(zone3_dev, 4),
            # 详细值
            'avg_candidate_sum': round(avg_cand_sum, 1),
            'avg_hist_sum': round(avg_hist_sum, 1),
            'avg_candidate_span': round(avg_cand_span, 1),
            'avg_hist_span': round(avg_hist_span, 1),
            'avg_candidate_big_ratio': round(avg_cand_big, 3),
            'avg_hist_big_ratio': round(avg_hist_big, 3),
            'avg_candidate_zone1': round(avg_cand_zone1, 3),
            'avg_hist_zone1': round(avg_hist_zone1, 3),
            'avg_candidate_zone3': round(avg_cand_zone3, 3),
            'avg_hist_zone3': round(avg_hist_zone3, 3),
        }

        print(f"[DLT-Fusion] 📊 偏差仪表盘: "
              f"和值偏差={sum_dev:+.2%} (候选{avg_cand_sum:.0f}/历史{avg_hist_sum:.0f}), "
              f"跨度偏差={span_dev:+.2%}, "
              f"大号比偏差={size_dev:+.2%}, "
              f"Z1区偏差={zone1_dev:+.2%}, Z3区偏差={zone3_dev:+.2%}")

        return dashboard

    def _recalibrate_confidence(self, candidates: List[Dict[str, Any]],
                                 dashboard: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        置信度重校准：基于偏差仪表盘调整最终评分。

        当某维度候选与历史分布偏差过大时，该维度可能采样不足，
        需要微调评分来覆盖被低估的方向。

        校准规则：
        - sum_deviation > +0.30 → 候选和值偏高，给低和值候选微加1%
        - sum_deviation < -0.30 → 候选和值偏低，给高和值候选微加1%
        - size_deviation > +0.30 → 大号偏多，给偏大号组合加1%
        - size_deviation < -0.30 → 小号偏多，给偏小号组合加1%
        """
        sum_dev = dashboard.get('sum_deviation', 0)
        size_dev = dashboard.get('size_deviation', 0)
        # P5: 三区偏差
        zone1_dev = dashboard.get('zone1_deviation', 0)
        zone3_dev = dashboard.get('zone3_deviation', 0)

        cal_count = 0
        zone_cal_count = 0
        for c in candidates:
            front = c.get('front', [])
            if not front:
                continue

            orig = c.get('final_score', c.get('base_score', 0.5))
            front_sum = sum(front)
            big_ratio = sum(1 for n in front if n >= 25) / 5.0
            small_ratio = sum(1 for n in front if n <= 12) / 5.0

            # 和值校准
            if sum_dev > 0.30 and front_sum < 90:
                c['final_score'] = orig * 1.01
                cal_count += 1
            elif sum_dev < -0.30 and front_sum > 110:
                c['final_score'] = orig * 1.01
                cal_count += 1

            # 大小号校准（原有）
            if size_dev > 0.30 and big_ratio <= 0.2:
                c['final_score'] = orig * 1.01
                cal_count += 1
            elif size_dev < -0.30 and big_ratio >= 0.6:
                c['final_score'] = orig * 1.01
                cal_count += 1

            # P5: 三区分布动态校准
            # zone1_dev > +0.30 → 候选小号区偏多 → 给中大号组合加分
            # zone3_dev > +0.30 → 候选大号区偏多 → 给小中号组合加分
            boost = 1.0
            if zone1_dev > 0.30 and big_ratio >= 0.4:
                # 候选小号过多了，给含大号(≥25)的加分
                boost *= 1.02
                zone_cal_count += 1
            elif zone1_dev < -0.30 and small_ratio >= 0.4:
                # 候选小号不足，给小号多的加分
                boost *= 1.02
                zone_cal_count += 1

            if zone3_dev > 0.30 and small_ratio >= 0.4:
                # 候选大号偏多，给小号多的加分
                boost *= 1.02
                zone_cal_count += 1
            elif zone3_dev < -0.30 and big_ratio >= 0.4:
                # 候选大号偏少，给大号多的加分
                boost *= 1.02
                zone_cal_count += 1

            if boost > 1.0:
                c['final_score'] = orig * boost
                cal_count += 1

        if cal_count > 0:
            msg = f"[DLT-Fusion] 🔄 置信度重校准: 共调整{cal_count}注"
            if zone_cal_count > 0:
                msg += f" (含三区校准{zone_cal_count}注)"
            print(msg)

        return candidates

    # ------------------------------------------------------------------
    # 🎲 命中概率计算（增强输出）
    # ------------------------------------------------------------------

    def _calc_probability(self, front: List[int], back: List[int]) -> Dict[str, float]:
        """
        计算候选号码的命中概率（经验加权评分）。
        输出百分比形式，不是真实概率（彩票号码不可预测）。
        用于候选之间的相对排序参考。
        """
        front_arr = np.array(front)
        back_arr = np.array(back)
        n = len(self.draws)
        recent_n = min(20, n)

        # 前区频率得分
        front_freq_prob = np.mean([
            sum(1 for f_, _ in self.draws if num in f_) / max(n, 1)
            for num in front_arr
        ])

        # 近期趋势得分（近20期）
        recent_draws = self.draws[-recent_n:] if len(self.draws) >= recent_n else self.draws
        front_recent_prob = np.mean([
            sum(1 for f_, _ in recent_draws if num in f_) / max(len(recent_draws), 1)
            for num in front_arr
        ])

        # 遗漏得分
        front_missing = []
        for num in front_arr:
            miss = n - sum(1 for f_, _ in self.draws if num in f_)
            front_missing.append(miss)
        front_missing_prob = min(np.mean(front_missing) / 100, 1.0)

        # 和值评分
        front_sum = np.sum(front_arr)
        sum_score = 1.0 if 80 <= front_sum <= 130 else max(0, 1 - abs(front_sum - 105) / 50)

        # 奇偶评分
        odd_count = np.sum(front_arr % 2)
        odd_score = 1.0 - abs(odd_count - 2.5) / 2.5

        # 跨度评分
        span = max(front_arr) - min(front_arr)
        span_score = 1.0 if 15 <= span <= 32 else max(0, 1 - abs(span - 24) / 20)

        # 后区
        back_freq_prob = np.mean([
            sum(1 for _, b_ in self.draws if num in b_) / max(n, 1)
            for num in back_arr
        ])
        back_recent_prob = np.mean([
            sum(1 for _, b_ in recent_draws if num in b_) / max(len(recent_draws), 1)
            for num in back_arr
        ])
        back_missing = []
        for num in back_arr:
            miss = n - sum(1 for _, b_ in self.draws if num in b_)
            back_missing.append(miss)
        back_missing_prob = min(np.mean(back_missing) / 100, 1.0)
        back_sum = np.sum(back_arr)
        back_sum_score = 1.0 if 3 <= back_sum <= 23 else max(0, 1 - abs(back_sum - 13) / 12)

        front_prob = (front_freq_prob * 0.15 + front_recent_prob * 0.25 +
                      front_missing_prob * 0.15 + sum_score * 0.20 +
                      odd_score * 0.10 + span_score * 0.15)
        back_prob = (back_freq_prob * 0.15 + back_recent_prob * 0.25 +
                     back_missing_prob * 0.20 + back_sum_score * 0.25 +
                     (1.0 if back_arr[0] != back_arr[1] else 0.5) * 0.15)
        combined_prob = front_prob * 0.70 + back_prob * 0.30

        return {
            'combined': round(combined_prob * 100, 2),
            'front': round(front_prob * 100, 2),
            'back': round(back_prob * 100, 2),
        }

    # ------------------------------------------------------------------
    # 📊 胆拖投注方案生成
    # ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  DLT多策略融合完全体 V3.0.0")
    print("=" * 60)

    data_path = data_dir()

    try:
        fusion = DLTFusionComplete(data_path)
    except Exception as e:
        print(f"[DLT-Fusion] 初始化失败: {e}")
        return

    # 测试各模块
    print("\n📊 各组推荐：")
    try:
        groups = fusion.get_group_recommendations()
        for gid, cands in groups.items():
            if cands:
                print(f"  组{gid}: {cands[0].front} | score={cands[0].total_score:.4f}")
    except Exception as e:
        print(f"  获取分组推荐失败: {e}")

    print("\n🎯 后区推荐：")
    try:
        back = fusion.get_back_recommendations()
        for i, b in enumerate(back[:5]):
            print(f"  方案{i+1}: {b}")
    except Exception as e:
        print(f"  获取后区推荐失败: {e}")

    print("\n🏆 最终融合推荐（Top 5）：")
    try:
        results = fusion.predict(top_n=5)
        for i, r in enumerate(results):
            print(f"  {i+1}. 前区{r['front']} 后区{r['back']} "
                  f"[{r['strategy_name']}] score={r['final_score']:.4f}")
    except Exception as e:
        print(f"  融合预测失败: {e}")

    print("\n📈 回测结果（最近50期）：")
    try:
        bt = fusion.backtest(50)
        if 'error' in bt:
            print(f"  {bt['error']}")
        else:
            print(f"  测试次数: {bt['n_tests']}")
            print(f"  平均前区命中: {bt['avg_front_hits']:.3f}")
            print(f"  平均后区命中: {bt['avg_back_hits']:.3f}")
            print(f"  前区命中分布: {bt['front_hit_distribution']}")
            print(f"  后区命中分布: {bt['back_hit_distribution']}")
    except Exception as e:
        print(f"  回测失败: {e}")

    print("\n📋 详细分析：")
    try:
        detailed = fusion.predict_with_details(top_n=3)
        print(f"  总记录: {detailed['total_records']}期")
        for i, pred in enumerate(detailed['predictions'][:3]):
            print(f"  推荐{i+1}: {pred['front']} + {pred['back']} "
                  f"(综合分: {pred['final_score']:.4f})")
    except Exception as e:
        print(f"  详细分析失败: {e}")

    print("\n" + "=" * 60)
    print("  运行完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
