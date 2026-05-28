#!/usr/bin/env python3
"""
DLT多策略融合完全体 V2.0
整合策略融合引擎 + 五池采样 + 后区融合 + 博弈论 + 遗传算法 + 数学过滤 + 统计分析
+ 隔期重号增强(SkipRepeatBooster) + 双期参考候选 + 后区隔期重号 + 智能重号惩罚
"""

import sys
import os
import os.path as _path
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict

# 数据文件路径（基于技能包目录自动定位）
def data_dir() -> str:
    return _path.join(_path.dirname(_path.abspath(__file__)), 'data', 'DLT历史数据_适配模型版.xlsx')


# 导入所有子模块
from dlt_predictor_upgraded import DLTPredictorUpgraded, load_dlt_data
from strategy_fusion_engine import StrategyFusionEngine
from five_pool_sampler_complete_final import FivePoolSampler
from dlt_back_fusion import BackZoneFusion
from modules.dlt_game_theory import DLTGameTheoryAnalyzer
from modules.dlt_genetic_optimizer import DLTGeneticOptimizer
from modules.dlt_math_filter import DLTMathFilter
from modules.dlt_statistics_analyzer import DLTStatisticsAnalyzer
from modules.dlt_pattern_recognizer import DLTPatternRecognizer, apply_pattern_boost, generate_pattern_diversity_pool


class DLTFusionComplete:
    """DLT多策略融合完全体 — 整合所有预测模块的统一入口"""

    def __init__(self, data_path: Optional[str] = None, auto_update: bool = True):
        if data_path is None:
            data_path = data_dir()
        self.data_path = data_path

        # 1. 加载数据（带fallback）
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
        self.pool_sampler = FivePoolSampler(self.draws)
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

        # 初始化遗传算法
        try:
            self.genetic.evolve(generations=50, verbose=False)
        except Exception as e:
            print(f"[DLT-Fusion] 遗传算法初始化: {e}")

        print(f"[DLT-Fusion] 初始化完成 | V1.1完全体")

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
            for j in range(len(df)):
                front = sorted([int(df.iloc[j][f'前区{i}']) for i in range(1, 6)])
                back = sorted([int(df.iloc[j][f'后区{i}']) for i in range(1, 3)])
                draws.append((front, back))
            if draws and draws[0][0][0] > draws[-1][0][0]:
                draws = list(reversed(draws))
            if draws:
                print(f"[DLT-Fusion] 直接读取Excel: {len(draws)}期")
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
            drift_strength = confidence

            if direction == 'up':
                # 重心在向高区移动：提升三区权重，降低一区权重
                adjustments['z1'] = max(0.6, 1.0 - drift_strength * 0.3)
                adjustments['z2'] = 1.0  # 二区不变
                adjustments['z3'] = min(1.5, 1.0 + drift_strength * 0.4)
            elif direction == 'down':
                # 重心在向低区移动：提升一区权重，降低三区权重
                adjustments['z1'] = min(1.5, 1.0 + drift_strength * 0.4)
                adjustments['z2'] = 1.0
                adjustments['z3'] = max(0.6, 1.0 - drift_strength * 0.3)

            if confidence > 0.5:
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

        boost_strength = min(confidence * 0.12, 0.08)  # max 8% score adjustment

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
            self.pool_sampler = FivePoolSampler(self.draws)
            self.back_fusion = BackZoneFusion(self.draws)
            self.genetic = DLTGeneticOptimizer(self.draws)
            self.stats = DLTStatisticsAnalyzer(self.draws)
            self.pattern_recognizer = DLTPatternRecognizer(self.draws)
            self.pattern_recognizer.build_distributions(window=500)
            self.genetic.evolve(generations=50, verbose=False)
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
        生成多种复式投注

        Args:
            bet_type: '6+3','7+2','7+3','8+2','8+3','9+3','9+4','9+6','all'
            n_per_type: 每种类型生成多少组

        Returns:
            Dict[bet_type, List[复式投注]]
        """
        bet_map = {
            '6+3': lambda: self.pool_sampler.generate_6_plus_3(n=n_per_type, strategy='game_theory'),
            '6+4': lambda: self.pool_sampler.generate_6_plus_4(n=n_per_type, strategy='game_theory'),
            '7+2': lambda: self.pool_sampler.generate_7_plus_2(n=n_per_type, strategy='mixed'),
            '7+3': lambda: self.pool_sampler.generate_7_plus_3(n=n_per_type, strategy='mixed'),
            '7+4': lambda: self.pool_sampler.generate_7_plus_4(n=n_per_type, strategy='balance'),
            '8+2': lambda: self.pool_sampler.generate_8_plus_2(n=n_per_type, strategy='hot'),
            '8+3': lambda: self.pool_sampler.generate_8_plus_3(n=n_per_type, strategy='mixed'),
            '8+4': lambda: self.pool_sampler.generate_8_plus_4(n=n_per_type, strategy='cold'),
            '8+5': lambda: self.pool_sampler.generate_8_plus_5(n=n_per_type, strategy='cold'),
            '9+3': lambda: self.pool_sampler.generate_9_plus_3(n=n_per_type, strategy='cold'),
            '9+4': lambda: self.pool_sampler.generate_9_plus_4(n=n_per_type, strategy='cold'),
            '9+6': lambda: self.pool_sampler.generate_9_plus_6(n=n_per_type, strategy='balance'),
        }

        if bet_type == 'all':
            results = {}
            for bt, fn in bet_map.items():
                try:
                    results[bt] = fn()
                except Exception:
                    results[bt] = []
            return results
        elif bet_type in bet_map:
            return {bet_type: bet_map[bet_type]()}
        else:
            return {'6+3': bet_map['6+3']()}

    def predict(self, top_n: int = 5, include_compound: bool = True) -> Dict[str, Any]:
        """主预测函数：触发预测时自动同步最新开奖数据，然后生成推荐"""
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

        # Step 2: 五池采样补充候选 + 模式池采样 + 双期重号参考候选
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
        if drift_info['drift_detected'] and drift_info['confidence'] >= 0.15:
            all_candidates = self._apply_drift_boost(all_candidates, drift_info)

        # 【方案A】隔期重号评分增强 — 候选与上上期匹配时加分
        try:
            all_candidates = self._apply_skip_repeat_boost(all_candidates)
        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 隔期重号评分跳过: {e}")

        # Step 7c: 重号惩罚——候选与上期重号≥3个时降分5%，避免热号过度堆叠
        # 【方案D】区分预期重号vs热号堆叠
        all_candidates = self._apply_repeat_penalty(all_candidates)

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

        # Step 10: 最终排名
        unique.sort(key=lambda x: x['final_score'], reverse=True)

        result = {
            'single_bets': unique[:top_n],
        }

        # 添加复式投注推荐
        if include_compound:
            compound = self.generate_compound_bets('all', n_per_type=2)
            result['compound_bets'] = compound

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

        from five_pool_sampler_complete_final import FivePoolSampler

        for i, actual in enumerate(test_draws):
            # 用历史数据生成各池
            hist = train_base + test_draws[:i] if i > 0 else train_base
            sampler = FivePoolSampler(hist)

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
        """五池采样候选"""
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
                    'strategy_name': 'FivePoolSampler',
                })
        except Exception as e:
            print(f"[DLT-Fusion] 五池采样失败: {e}")

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

    def _compute_final_scores(self, candidates: List[Dict[str, Any]]) -> None:
        """计算综合评分"""
        for c in candidates:
            # 综合评分: base*0.4 + gt*0.3 + genetic*0.3
            c['final_score'] = c['base_score'] * 0.4 + c['gt_score'] * 0.3 + c['genetic_score'] * 0.3

    def _deduplicate_and_assign_back(
        self,
        candidates: List[Dict[str, Any]],
        back_recs: List[List[int]]
    ) -> List[Dict[str, Any]]:
        """去重并分配后区"""
        seen = set()
        unique = []

        for c in candidates:
            k = tuple(c['front'])
            if k not in seen:
                seen.add(k)
                # 分配后区
                if back_recs and len(unique) < len(back_recs):
                    c['back'] = back_recs[len(unique) % len(back_recs)]
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

        增强策略：
        - 前区与上上期匹配≥2个：最终评分×1.15
        - 后区与上上期匹配≥1个：最终评分×1.10
        - 前后区同时匹配（前≥2且后≥1）：最终评分×1.20
        - 前区匹配≥3个（含上上期和上期的差异部分）：×1.08

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
                multiplier = 1.20
                reasons.append(f'前后隔期双匹配(前{front_skip_overlap}+后{back_skip_overlap})')
                strong_boost_count += 1
            elif front_skip_overlap >= 3:
                # 前区高度匹配上上期：中等增强
                multiplier = 1.15
                reasons.append(f'前区隔期高匹配({front_skip_overlap}个)')
            elif front_skip_overlap >= 2:
                # 前区匹配≥2：基础增强
                multiplier = 1.12
                reasons.append(f'前区隔期匹配({front_skip_overlap}个)')
            elif back_skip_overlap >= 1:
                # 后区单独匹配≥1：轻度增强
                multiplier = 1.08
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
        智能重号惩罚（方案D升级版）：区分预期重号与热号堆叠。

        核心改进：
        1. 单号码高概率重号（如上期开出的34在26058延续出现）→ 不应惩罚
        2. 多个热号无理由堆叠 → 惩罚

        判定逻辑：
        - 计算候选与上期的每个重叠号码的"冷热度"：
          - 冷号重号（该号码近期未密集出现）：预期重号，不惩罚
          - 热号重号（该号码近期已频繁出现）：热号堆叠，惩罚
        - 重叠数≥3且其中热号重号占多数（≥2个热号）→ 5%折扣
        - 重叠数≥3但以冷号为主（≤1个热号）→ 视为正常趋势，不打折
        - 重叠数为3中恰好包含1-2个"间隔性出现"的号码时，做中性处理

        冷热度判定：号码在最近10期出现≥4次 = 热号（过热范围）
        """
        if len(self.draws) < 2:
            return candidates

        latest_front = set(self.draws[-1][0])

        # 计算最近10期每个号码的出现频率
        recent_window = 10
        recent_front_all = []
        for i in range(max(0, len(self.draws) - recent_window), len(self.draws)):
            recent_front_all.extend(self.draws[i][0])
        num_freq = Counter(recent_front_all)
        # 热号阈值：最近10期出现≥4次
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

            if hot_overlaps >= 2:
                # 热号堆叠（≥2个热号与上期重复）：5%折扣
                c['final_score'] = orig * 0.95
                penalty_count += 1
                if penalty_count <= 3:
                    print(f"[DLT-Fusion] 🔽 热号堆叠惩罚: 前区{c.get('front', [])} "
                          f"重叠{overlap}个(热{hot_overlaps}/冷{cold_overlaps}) "
                          f"(score: {orig:.4f}→{c['final_score']:.4f})")
            elif cold_overlaps >= 2:
                # 冷号/间隔性重号为主：视为趋势延续，小幅加分鼓励
                c['final_score'] = orig * 1.03
                neutral_count += 1
                if neutral_count <= 3:
                    print(f"[DLT-Fusion] ✅ 趋势延续识别: 前区{c.get('front', [])} "
                          f"重叠{overlap}个(热{hot_overlaps}/冷{cold_overlaps}) "
                          f"(score: {orig:.4f}→{c['final_score']:.4f})")
            # else: 混合型（1热1冷等），不做调整

        if penalty_count > 0:
            print(f"[DLT-Fusion] 🔽 重号惩罚(方案D)合计: {penalty_count}注受折扣")
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
        """获取最近一期期号"""
        try:
            if hasattr(self.predictor, 'data') and self.predictor.data is not None:
                periods = self.predictor.data.get('periods', [])
                if periods:
                    return str(periods[-1]) if not isinstance(periods[-1], int) else str(periods[-1])
            if hasattr(self.draws[0][0], '__len__') and len(self.draws) > 0:
                return None  # 无法从draws直接获取期号
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# 入口函数
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  DLT多策略融合完全体 V1.0")
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
