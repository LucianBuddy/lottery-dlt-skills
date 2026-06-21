#!/usr/bin/env python3
"""
DLT后区多维度融合策略模块
针对后区12选2的特殊性优化

核心特点：
1. 后区只有12个号码，冷热评分窗口更短（热:10期, 冷:20期）
2. 大小比1:1是核心约束（允许浮动）
3. 四池融合：热号40% + 冷号25% + 均衡池25% + 博弈论10%
4. 复用DLTFeatureExtractor的13维评分（策略融合引擎）
5. 复用MultiPoolSampler的多池采样
6. 【冷三联因子】后区过热反转检测：连续2期热号组合后强制偏移冷号30%

作者: 贾维斯 (JARVIS)
日期: 2026-04-07
"""

import sys
import os
import random
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

# 路径设置，确保可以导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 复用现有的多池采样器
from five_pool_sampler_complete_final import MultiPoolSampler

# 复用策略融合引擎中的13维特征提取器（替代已废弃的 dlt_strategy_fusion_v2.FeatureScorer）
from strategy_fusion_engine import DLTFeatureExtractor

try:
    from modules.dlt_game_theory import DLTGameTheoryAnalyzer
except ImportError:
    DLTGameTheoryAnalyzer = None


# ============================================================
# 常量定义
# ============================================================

# 后区号码范围
BACK_MIN = 1
BACK_MAX = 12

# 后区大小划分：1-6小，7-12大
BACK_SIZE = {
    'small': set(range(1, 7)),    # 1-6小
    'large': set(range(7, 13))     # 7-12大
}

# 后区热冷号定义窗口
BACK_HOT_WINDOW = 10   # 热号窗口：最近10期
BACK_HOT_THRESHOLD = 3  # 热号阈值：出现≥3次
BACK_COLD_WINDOW = 20   # 冷号窗口：最近20期
BACK_COLD_THRESHOLD = 1  # 冷号阈值：出现≤1次


class BackZoneFusion:
    """
    后区多维度融合策略
    
    基于多池×13维融合框架，针对后区12选2的特殊性优化。
    
    策略权重：
    - 热号池：40% （最近10期出现≥3次）
    - 冷号池：25% （最近20期出现≤1次，回补机会）
    - 均衡池：25% （热冷各半）
    - 博弈论池：10%（大众偏好回避度）
    - 【冷三联因子】后区过热反转：连续2期热号组合时强制偏移冷号30%
    
    大小比约束：
    - 核心：1大1小（1:1）
    - 允许浮动：2大或2小（2:0或0:2）
    
    使用方法：
        draws = [(front_nums, back_nums), ...]  # 历史开奖
        fusion = BackZoneFusion(draws)
        recommendations = fusion.generate_recommendations(n=5)
    """

    # 后区特别策略（最终策略设计）
    BACK_STRATEGY: Dict[str, Any] = {
        'size_ratio': (1, 1),       # 1:1大小比（核心）
        'hot_weight': 0.40,          # 热号权重
        'cold_weight': 0.25,         # 冷号权重（回补机会）
        'balance_weight': 0.25,      # 均衡池权重
        'game_weight': 0.10,         # 博弈论权重
    }

    # 后区13维评分权重（默认等权，可外部传入）
    DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
        'hot_cold': 1.0,
        'odd_even': 0.8,
        'consecutive': 0.5,
        'repeat': 0.7,
        'adjacent': 0.6,
        'sum': 0.7,
        'prime': 0.8,
        'missing_bayesian': 1.0,
        'size': 0.9,
        'zone': 0.6,
        'ac': 0.5,
        'range': 0.5,
        'cooccurrence': 0.7,
    }

    def __init__(self, draws: List[Tuple[List[int], List[int]]]):
        """
        初始化后区融合器
        
        Args:
            draws: 历史开奖数据，格式为[(前区5个号码, 后区2个号码), ...]
        """
        self.draws = draws
        self.back_draws = [d[1] for d in draws]
        self.n = len(draws)
        
        # 初始化特征提取器（复用DLTFeatureExtractor的13维特征提取能力）
        self.feature_extractor = DLTFeatureExtractor(draws, window=30)
        
        # 初始化多池采样器（复用MultiPoolSampler）
        if MultiPoolSampler is not None:
            self.pool_sampler = MultiPoolSampler(draws)
        else:
            self.pool_sampler = None
        
        # 初始化博弈论分析器
        if DLTGameTheoryAnalyzer is not None:
            self.game_analyzer = DLTGameTheoryAnalyzer()
        else:
            self.game_analyzer = None
        
        # 预计算后区统计数据
        self._compute_back_stats()
        
        # 预计算热冷池
        self._compute_hot_cold_pools()
        
        # 预计算各池评分
        self._compute_pool_scores()

    def _compute_back_stats(self) -> None:
        """预计算后区基础统计"""
        self.back_freq: Dict[int, int] = defaultdict(int)
        for back in self.back_draws:
            for n in back:
                self.back_freq[n] += 1
        
        # 遗漏次数
        self.back_missing: Dict[int, int] = defaultdict(int)
        for back in self.back_draws:
            for n in range(BACK_MIN, BACK_MAX + 1):
                if n not in back:
                    self.back_missing[n] += 1

    def _compute_hot_cold_pools(self) -> None:
        """预计算热冷均衡池"""
        # 统计最近BACK_HOT_WINDOW期的出现次数
        recent = self.back_draws[-BACK_HOT_WINDOW:] if len(self.back_draws) > BACK_HOT_WINDOW else self.back_draws
        recent_freq: Dict[int, int] = defaultdict(int)
        for back in recent:
            for n in back:
                recent_freq[n] += 1
        
        # 热号池：最近10期出现≥3次
        self.hot_pool: List[int] = [
            n for n in range(BACK_MIN, BACK_MAX + 1)
            if recent_freq.get(n, 0) >= BACK_HOT_THRESHOLD
        ]
        
        # 冷号池：最近20期出现≤1次
        cold_recent = self.back_draws[-BACK_COLD_WINDOW:] if len(self.back_draws) > BACK_COLD_WINDOW else self.back_draws
        cold_freq: Dict[int, int] = defaultdict(int)
        for back in cold_recent:
            for n in back:
                cold_freq[n] += 1
        
        self.cold_pool: List[int] = [
            n for n in range(BACK_MIN, BACK_MAX + 1)
            if cold_freq.get(n, 0) <= BACK_COLD_THRESHOLD
        ]
        
        # 均衡池：热冷各半
        half = 6 // 2
        hot_part = self.hot_pool[:half] if len(self.hot_pool) > half else self.hot_pool
        cold_part = self.cold_pool[:half] if len(self.cold_pool) > half else self.cold_pool
        self.balance_pool: List[int] = list(set(hot_part + cold_part))

    def _compute_pool_scores(self) -> None:
        """
        预计算每个池的评分贡献
        为每个号码在各个池中的评分（用于加权融合）
        """
        all_nums = list(range(BACK_MIN, BACK_MAX + 1))
        
        # 热号评分（最近10期频率归一化）
        recent = self.back_draws[-BACK_HOT_WINDOW:] if len(self.back_draws) > BACK_HOT_WINDOW else self.back_draws
        freq: Dict[int, int] = defaultdict(int)
        for back in recent:
            for n in back:
                freq[n] += 1
        max_f = max(freq.values()) if freq else 1
        min_f = min(freq.values()) if freq else 0
        
        self.hot_scores: Dict[int, float] = {}
        for n in all_nums:
            if max_f == min_f:
                self.hot_scores[n] = 0.5
            else:
                self.hot_scores[n] = (freq[n] - min_f) / (max_f - min_f)
        
        # 冷号评分（遗漏次数归一化）
        miss_vals = [self.back_missing[n] for n in all_nums]
        max_m = max(miss_vals) if miss_vals else 1
        min_m = min(miss_vals) if miss_vals else 0
        
        self.cold_scores: Dict[int, float] = {}
        for n in all_nums:
            if max_m == min_m:
                self.cold_scores[n] = 0.5
            else:
                self.cold_scores[n] = (self.back_missing[n] - min_m) / (max_m - min_m)
        
        # 均衡评分（热冷评分的均值）
        self.balance_scores: Dict[int, float] = {}
        for n in all_nums:
            self.balance_scores[n] = (self.hot_scores.get(n, 0.5) + self.cold_scores.get(n, 0.5)) / 2.0
        
        # 博弈论评分（简化版，用于后区）
        self.game_scores: Dict[int, float] = {}
        if self.game_analyzer is not None:
            for n in all_nums:
                # 博弈论对单号码评分：避开热门偏好
                score = 0.5
                # 避开1-3超小号（大众偏好）
                if n <= 3:
                    score += 0.2
                # 避开10-12超大号（冷门）
                if n >= 10:
                    score += 0.15
                # 中间区域较均衡
                self.game_scores[n] = min(score, 1.0)
        else:
            self.game_scores = {n: 0.5 for n in all_nums}

    def _get_combo_level_scores(self, combo: List[int]) -> Dict[str, float]:
        """
        计算后区组合级特征评分（ac/range/consecutive）
        这些特征无法在单号码级评估，必须在组合级计算。
        """
        n1, n2 = combo[0], combo[1]

        # AC值：后区只用2个号码，AC值固定为两数之差
        ac_val = abs(n1 - n2)
        # 后区AC值归一化：最大差11（12-1），最小差1
        ac_score = (ac_val - 1) / 10.0 if ac_val > 0 else 0.0

        # 跨度（range）：后区两码之差
        range_val = abs(n1 - n2)
        # 归一化：最大11，最小1
        range_score = (range_val - 1) / 10.0 if range_val > 0 else 0.0

        # 连号：后区连号的概率
        consecutive_score = 1.0 if abs(n1 - n2) == 1 else 0.0

        return {
            'ac': round(ac_score, 4),
            'range': round(range_score, 4),
            'consecutive': round(consecutive_score, 4),
        }

    def get_back_scores(self, zone: str = 'back') -> Dict[str, Dict[int, float]]:
        """
        获取后区13维评分（号码级）
        
        维度包括：hot_cold, odd_even, consecutive, repeat,
        adjacent, sum, prime, missing_bayesian, size,
        zone, ac, range, cooccurrence
        
        注意：consecutive/ac/range 是组合级特征，
        在号码级返回空（由 _get_combo_level_scores 在组合级计算）。
        
        Args:
            zone: 区域类型，默认'back'
            
        Returns:
            Dict[str, Dict[int, float]]: {维度名: {号码: 分数}}
                组合级维度返回空 dict {}
        """
        all_nums = list(range(BACK_MIN, BACK_MAX + 1))
        fe = self.feature_extractor
        result: Dict[str, Dict[int, float]] = {}

        # ── DLTFeatureExtractor 提供的方法 ──
        result['hot_cold'] = {n: fe.get_hot_score(n, 'back') for n in all_nums}
        result['repeat'] = {n: fe.get_repeat_score(n, 'back') for n in all_nums}
        result['adjacent'] = {n: fe.get_adjacent_score(n, 'back') for n in all_nums}
        result['prime'] = {n: fe.get_prime_score(n, 'back') for n in all_nums}
        result['missing_bayesian'] = {n: fe.get_missing_score(n, 'back') for n in all_nums}
        result['cooccurrence'] = {n: fe.get_cooc_score(n, 'back') for n in all_nums}

        # ── 内联计算 ──
        # 奇偶：偶数=1，奇数=0
        result['odd_even'] = {n: 1.0 if n % 2 == 0 else 0.0 for n in all_nums}
        # 和值归一化：1→0.0, 12→1.0
        result['sum'] = {n: (n - 1) / 11.0 for n in all_nums}
        # 大小：小号(1-6)=0, 大号(7-12)=1
        result['size'] = {n: 1.0 if n >= 7 else 0.0 for n in all_nums}
        result['zone'] = result['size']  # 后区区间=大小

        # ── 组合级特征：号码级占位空dict ──
        result['ac'] = {}
        result['range'] = {}
        result['consecutive'] = {}

        return result

    def apply_size_constraint(self, candidates: List[List[int]]) -> List[List[int]]:
        """
        应用后区大小比约束：1大1小或2大或2小（允许浮动）
        
        后区大小划分：
        - 小号：1-6
        - 大号：7-12
        
        约束规则：
        - 核心要求：1大1小（1:1）
        - 允许浮动：2大0小 或 0大2小
        
        Args:
            candidates: 候选后区号码对列表
            
        Returns:
            List[List[int]]: 满足大小约束的号码对
        """
        valid = []
        for combo in candidates:
            if len(combo) != 2:
                continue
            
            small_count = sum(1 for n in combo if n in BACK_SIZE['small'])
            large_count = sum(1 for n in combo if n in BACK_SIZE['large'])
            
            # 允许：1:1, 2:0, 0:2（但总和必须为2）
            if small_count + large_count == 2:
                valid.append(sorted(combo))
        
        return valid

    def generate_back_candidates(self, n: int = 50) -> List[List[int]]:
        """
        生成后区候选组合（2个号码）
        
        使用四池加权采样生成候选对：
        1. 按四池权重对每个号码进行加权评分
        2. 选取评分最高的候选号码
        3. 配对生成所有可能的2号码组合
        4. 应用大小比约束过滤
        
        Args:
            n: 目标候选数量（默认50）
            
        Returns:
            List[List[int]]: 后区候选号码对列表
        """
        all_nums = list(range(BACK_MIN, BACK_MAX + 1))
        
        # 计算每个号码的综合评分
        combined_scores: Dict[int, float] = {}
        sw = self.BACK_STRATEGY
        
        for num in all_nums:
            score = (
                sw['hot_weight'] * self.hot_scores.get(num, 0.5) +
                sw['cold_weight'] * self.cold_scores.get(num, 0.5) +
                sw['balance_weight'] * self.balance_scores.get(num, 0.5) +
                sw['game_weight'] * self.game_scores.get(num, 0.5)
            )
            combined_scores[num] = score
        
        # 按评分排序，选取top候选
        sorted_nums = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_nums = [num for num, _ in sorted_nums[:max(n, 12)]]
        
        # 补充随机号码确保覆盖
        remaining = [num for num in all_nums if num not in top_nums]
        random.shuffle(remaining)
        candidates_pool = top_nums + remaining[:max(0, min(n, 12) - len(top_nums))]
        
        # 生成所有两两组合
        all_combos: List[Tuple[int, int]] = []
        for i, n1 in enumerate(candidates_pool):
            for n2 in candidates_pool[i + 1:]:
                all_combos.append((n1, n2))
        
        # 去重并应用大小约束
        seen: set = set()
        valid_combos: List[List[int]] = []
        for c in all_combos:
            key = tuple(sorted(c))
            if key in seen:
                continue
            seen.add(key)
            combo = sorted(c)
            small_count = sum(1 for num in combo if num in BACK_SIZE['small'])
            large_count = sum(1 for num in combo if num in BACK_SIZE['large'])
            if small_count + large_count == 2:
                valid_combos.append(combo)
        
        # 按综合评分排序
        def combo_score(combo: List[int]) -> float:
            return combined_scores.get(combo[0], 0) + combined_scores.get(combo[1], 0)
        
        valid_combos.sort(key=combo_score, reverse=True)
        
        return valid_combos[:n]

    def score_back_combo(self, combo: List[int]) -> Dict[str, float]:
        """
        对后区组合进行多维度评分
        
        评估维度：
        - 基础池：hot/cold/balance/game_score
        - 统计特征：size_score（大小比1:1得满分）
        - 组合级特征：ac_score（算术复杂度）、range_score（跨度）、
          consecutive_score（连号）
        - fusion_score: 综合融合评分（基础池加权）
        
        Args:
            combo: 后区号码对 [n1, n2]
            
        Returns:
            Dict[str, float]: {维度名: 分数}
        """
        n1, n2 = combo[0], combo[1]
        
        # 基础池评分
        hot = (self.hot_scores.get(n1, 0.5) + self.hot_scores.get(n2, 0.5)) / 2.0
        cold = (self.cold_scores.get(n1, 0.5) + self.cold_scores.get(n2, 0.5)) / 2.0
        balance = (self.balance_scores.get(n1, 0.5) + self.balance_scores.get(n2, 0.5)) / 2.0
        game = (self.game_scores.get(n1, 0.5) + self.game_scores.get(n2, 0.5)) / 2.0
        
        # 大小比评分
        small_count = sum(1 for n in combo if n in BACK_SIZE['small'])
        large_count = sum(1 for n in combo if n in BACK_SIZE['large'])
        
        if small_count == 1 and large_count == 1:
            size_score = 1.0   # 1:1 完美
        elif small_count in (0, 2) or large_count in (0, 2):
            size_score = 0.7   # 允许的浮动
        else:
            size_score = 0.3
        
        # 组合级特征评分（AC值/跨度/连号）
        combo_scores = self._get_combo_level_scores(combo)
        
        # 综合融合评分
        sw = self.BACK_STRATEGY
        fusion = (
            sw['hot_weight'] * hot +
            sw['cold_weight'] * cold +
            sw['balance_weight'] * balance +
            sw['game_weight'] * game
        )
        
        result = {
            'hot_score': round(hot, 4),
            'cold_score': round(cold, 4),
            'balance_score': round(balance, 4),
            'game_score': round(game, 4),
            'size_score': round(size_score, 4),
            'fusion_score': round(fusion, 4),
        }
        result.update(combo_scores)
        return result

    def select_back_pair(
        self,
        fused_scores: Dict[int, float],
        strategy: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        """
        根据融合评分选择后区号码对
        
        策略权重：热号40% + 冷号25% + 均衡25% + 博弈10%
        必须满足1:1大小比（允许浮动）
        
        Args:
            fused_scores: 融合后的号码评分 {号码: 分数}
            strategy: 自定义策略权重（覆盖默认）
            
        Returns:
            List[int]: 选中的后区号码对
        """
        sw = strategy if strategy is not None else self.BACK_STRATEGY
        
        # 生成候选
        candidates = self.generate_back_candidates(n=50)
        
        # 应用大小约束过滤
        candidates = self.apply_size_constraint(candidates)
        
        if not candidates:
            # 如果没有满足约束的，随机返回一对
            nums = list(range(BACK_MIN, BACK_MAX + 1))
            random.shuffle(nums)
            return sorted(nums[:2])
        
        # 对每个候选评分
        scored_candidates: List[Tuple[List[int], float, Dict[str, float]]] = []
        for combo in candidates:
            scores = self.score_back_combo(combo)
            fusion = scores['fusion_score']
            
            # 如果满足1:1大小比，加成
            small_count = sum(1 for n in combo if n in BACK_SIZE['small'])
            if small_count == 1:
                fusion += 0.1   # 1:1加成
            
            scored_candidates.append((combo, fusion, scores))
        
        # 按融合评分排序
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 返回最优号码对
        best_combo = scored_candidates[0][0]
        return sorted(best_combo)

    def fuse_back_dimensions(
        self,
        dimension_weights: Optional[Dict[str, float]] = None
    ) -> Dict[int, float]:
        """
        后区13维加权融合
        
        从DLTFeatureExtractor获取后区号码级10维评分，按权重加权融合
        得到每个号码的综合评分（组合级ac/range/consecutive不在此计算）
        
        Args:
            dimension_weights: 各维度权重，
                             默认为DEFAULT_DIMENSION_WEIGHTS
        
        Returns:
            Dict[int, float]: {号码: 融合评分}
        """
        weights = dimension_weights if dimension_weights else self.DEFAULT_DIMENSION_WEIGHTS

        # 获取13维评分
        dim_scores = self.get_back_scores(zone='back')

        # 过滤出有数据的号码级维度（排除组合级的 ac/range/consecutive）
        valid_dim_scores = {k: v for k, v in dim_scores.items() if v}

        if not valid_dim_scores:
            # 降级：使用四池融合评分
            all_nums = list(range(BACK_MIN, BACK_MAX + 1))
            sw = self.BACK_STRATEGY
            return {
                n: (
                    sw['hot_weight'] * self.hot_scores.get(n, 0.5) +
                    sw['cold_weight'] * self.cold_scores.get(n, 0.5) +
                    sw['balance_weight'] * self.balance_scores.get(n, 0.5) +
                    sw['game_weight'] * self.game_scores.get(n, 0.5)
                )
                for n in all_nums
            }

        # 归一化各维度评分并加权融合
        all_nums = list(range(BACK_MIN, BACK_MAX + 1))
        fused: Dict[int, float] = {n: 0.0 for n in all_nums}

        total_weight = sum(weights.get(dim, 0) for dim in valid_dim_scores)
        if total_weight == 0:
            return {n: 0.5 for n in all_nums}

        for dim_name, scores in valid_dim_scores.items():
            w = weights.get(dim_name, 0)
            if w == 0:
                continue
            for n in all_nums:
                fused[n] += (w / total_weight) * scores.get(n, 0.5)

        return fused

    # ==================================================================
    # 后区推荐生成（主入口）
    # ==================================================================

    def generate_recommendations(
        self,
        n: int = 5
    ) -> List[Tuple[List[int], Dict[str, float]]]:
        """
        生成n个后区推荐对+各维度评分详情

        【方案C】集成隔期重号评分：上上期后区号码加权重。
        【冷三联因子】后区过热反转：连续2期热号组合时强制偏移冷号。
        
        返回每个推荐号码对的详细评分，包括：
        - 基础池评分：hot_score, cold_score, balance_score, game_score
        - 大小比评分：size_score
        - 综合融合评分：fusion_score
        - 各池入选情况：in_hot, in_cold, in_balance
        - 13维融合评分（号码级10维 + 组合级3维）
        
        Args:
            n: 推荐数量，默认5
        
        Returns:
            List[Tuple[List[int], Dict[str, float]]]:
            [(后区号码对, {维度: 分数, ...}), ...]
        """
        # 获取融合后的号码评分
        fused_scores = self.fuse_back_dimensions()

        # 【方案C】获取隔期重号评分
        skip_repeat_scores = self._compute_skip_repeat_back_scores()

        # 【冷三联因子】检测后区过热反转信号
        cold_force_scores = self._detect_back_hot_cascade()

        # 生成候选并筛选
        candidates = self.generate_back_candidates(n=100)
        candidates = self.apply_size_constraint(candidates)

        if not candidates:
            # 降级：随机生成
            nums = list(range(BACK_MIN, BACK_MAX + 1))
            random.shuffle(nums)
            result = []
            for i in range(min(n, 6)):
                combo = sorted(nums[i * 2:(i + 1) * 2])
                scores = self.score_back_combo(combo)
                scores['in_hot'] = combo[0] in self.hot_pool or combo[1] in self.hot_pool
                scores['in_cold'] = combo[0] in self.cold_pool or combo[1] in self.cold_pool
                scores['in_balance'] = combo[0] in self.balance_pool or combo[1] in self.balance_pool
                scores['fused_score'] = fused_scores.get(combo[0], 0.5) + fused_scores.get(combo[1], 0.5)
                result.append((combo, scores))
            return result[:n]

        # 对候选评分并排序
        scored: List[Tuple[List[int], Dict[str, float], float]] = []
        for combo in candidates:
            scores = self.score_back_combo(combo)

            # 补充入选池信息
            scores['in_hot'] = combo[0] in self.hot_pool or combo[1] in self.hot_pool
            scores['in_cold'] = combo[0] in self.cold_pool or combo[1] in self.cold_pool
            scores['in_balance'] = combo[0] in self.balance_pool or combo[1] in self.balance_pool

            # 融合13维评分
            scores['fused_score'] = (
                fused_scores.get(combo[0], 0.5) + fused_scores.get(combo[1], 0.5)
            ) / 2.0

            # 综合最终评分（池融合 + 尺寸融合）
            final_score = (
                0.7 * scores['fusion_score'] +
                0.3 * scores['fused_score']
            )

            # 【方案C】隔期重号修正：上上期后区号码加权
            skip_boost = 0.0
            for n in combo:
                skip_boost += skip_repeat_scores.get(n, 0.0)
            skip_boost = skip_boost / 2.0  # 归一化到0~1
            if skip_boost > 0:
                # 隔期重号作为独立加分维度，权重10%
                final_score = final_score * 0.9 + skip_boost * 0.1
                scores['skip_repeat_score'] = round(skip_boost, 4)

            # 【冷三联因子】后区过热反转补偿
            cold_force = 0.0
            for n in combo:
                cold_force += cold_force_scores.get(n, 0.0)
            cold_force = cold_force / 2.0  # 归一化到0~1
            if cold_force > 0:
                # 冷三联因子权重30%（过热时强制偏移）
                final_score = final_score * 0.70 + cold_force * 0.30
                scores['cold_force_score'] = round(cold_force, 4)
                if cold_force >= 0.6:
                    print(f"[Back-Fusion] ❄️ 冷三联因子激活: {combo} "
                          f"force={cold_force:.2f}")

            scores['final_score'] = round(final_score, 4)

            scored.append((combo, scores, final_score))

        # 按最终评分排序
        scored.sort(key=lambda x: x[2], reverse=True)

        # 返回top-n
        result: List[Tuple[List[int], Dict[str, float]]] = []
        seen: set = set()
        for combo, scores, _ in scored:
            key = tuple(combo)
            if key in seen:
                continue
            seen.add(key)
            result.append((combo, scores))
            if len(result) >= n:
                break

        # 【V3.1.4-④】后区全偶/全奇路径注入
        # 当常规候选缺乏非均衡奇偶组合时，主动注入
        all_even = [c for c in scored if c[0][0] % 2 == 0 and c[0][1] % 2 == 0]
        all_odd = [c for c in scored if c[0][0] % 2 == 1 and c[0][1] % 2 == 1]
        if not any(c[0][0] % 2 == 0 and c[0][1] % 2 == 0 for c in result):
            # 无全偶组合，从候选中选取最佳全偶注入
            for combo, scores, _ in all_even:
                if tuple(combo) not in seen:
                    seen.add(tuple(combo))
                    result.append((combo, scores))
                    if len(result) >= n:
                        break
                    break  # 只注入1个
        if not any(c[0][0] % 2 == 1 and c[0][1] % 2 == 1 for c in result):
            # 无全奇组合，注入1个
            for combo, scores, _ in all_odd:
                if tuple(combo) not in seen:
                    seen.add(tuple(combo))
                    result.append((combo, scores))
                    if len(result) >= n:
                        break
                    break  # 只注入1个
        # 若仍缺乏全偶/全奇，从候选池生成并注入
        if not any(c[0][0] % 2 == 0 and c[0][1] % 2 == 0 for c in result):
            # 强制构建一个全偶组合
            evens = [n for n in range(2, 13, 2)]  # 2,4,6,8,10,12
            # 选择最近出现最少的一对全偶号码
            even_freq = {n: self.back_freq.get(n, 0) for n in evens}
            even_sorted = sorted(evens, key=lambda n: even_freq[n])
            for i, n1 in enumerate(even_sorted):
                for n2 in even_sorted[i+1:]:
                    combo = sorted([n1, n2])
                    if tuple(combo) not in seen:
                        seen.add(tuple(combo))
                        result.append((combo, {'fusion_score': 0.5, 'final_score': 0.5,
                                               'source': '全偶注入'}))
                        break
                if len(result) > len([c for c in result if c[0][0] % 2 == 0 and c[0][1] % 2 == 0]):
                    break
        if not any(c[0][0] % 2 == 1 and c[0][1] % 2 == 1 for c in result):
            odds = [n for n in range(1, 13, 2)]  # 1,3,5,7,9,11
            odd_freq = {n: self.back_freq.get(n, 0) for n in odds}
            odd_sorted = sorted(odds, key=lambda n: odd_freq[n])
            for i, n1 in enumerate(odd_sorted):
                for n2 in odd_sorted[i+1:]:
                    combo = sorted([n1, n2])
                    if tuple(combo) not in seen:
                        seen.add(tuple(combo))
                        result.append((combo, {'fusion_score': 0.5, 'final_score': 0.5,
                                               'source': '全奇注入'}))
                        break
                if len(result) > len([c for c in result if c[0][0] % 2 == 1 and c[0][1] % 2 == 1]):
                    break

        return result[:n]

    # ------------------------------------------------------------------
    # 【方案C】后区隔期重号模式检测
    # ------------------------------------------------------------------

    def _compute_skip_repeat_back_scores(self) -> Dict[int, float]:
        """
        计算后区隔期重号评分。

        核心逻辑：
        - 检查draws[-2](上上期)的后区号码
        - 如果这两个号码在当前期候选中有隔期回归信号，给予额外权重
        - 适用范围：历史中后区隔期重号（同一组号码隔期重现）的常见模式

        Returns:
            Dict[int, float]: {后区号码: 隔期重号评分(0~1)}
        """
        scores: Dict[int, float] = {n: 0.0 for n in range(1, 13)}

        if len(self.draws) < 3:
            return scores

        skip_back = set(self.draws[-2][1])  # 上上期后区号码
        prev_back = set(self.draws[-1][1])  # 上期后区号码

        # 只有上上期和上期后区不同时，隔期回归才值得关注
        # 如果已经连续两期相同，第三期回归概率增加
        if skip_back == prev_back:
            # 上上期=上期相同：第三期继续该组号码的概率很低
            # 但仍给轻微权重以防连续
            for n in skip_back:
                scores[n] = 0.3
        else:
            # 上上期≠上期：上上期号码隔期回归的概率较高
            for n in skip_back:
                # 上期没有出现的上上期号码，隔期回归概率最高
                if n not in prev_back:
                    scores[n] = 0.85
                else:
                    # 上期也出现了（连续三期出现的号码），概率较低
                    scores[n] = 0.25

            # 上上期后区的对称号/邻号也有一定概率回归
            for n in skip_back:
                # 邻号
                for adj in [n - 1, n + 1]:
                    if 1 <= adj <= 12 and adj not in skip_back and adj not in prev_back:
                        scores[adj] = max(scores[adj], 0.4)

        return scores

    # ==================================================================
    # 【冷三联因子】后区过热反转检测
    # ==================================================================

    def _detect_back_hot_cascade(self) -> Dict[int, float]:
        """
        冷三联因子：后区过热反转检测。

        核心逻辑：
        - 检查最近2期的后区号码是否均为"热号组合"
        - 热号组合定义：两个号码都是当前热号池中的号码
        - 当连续2期后区均为热号组合时，称为"后区过热"
        - 过热状态下，下一期冷号回补概率大幅增加
        - 对当前冷号池中号码给予额外权重（30%强制偏移）

        Returns:
            Dict[int, float]: {后区号码: 冷三联因子评分(0~1)}
                冷号池中的号码获得0.6~0.9评分，热号池中的号码获得0.1~0.3
        """
        cold_force: Dict[int, float] = {n: 0.0 for n in range(1, 13)}

        if len(self.draws) < 4:
            return cold_force

        # 检查最近2期后区是否均命中热号池
        def is_hot_combo(combo: List[int]) -> bool:
            """判断一个后区组合是否两个号码都来自热号池"""
            hot_count = sum(1 for n in combo if n in self.hot_pool)
            return hot_count >= 2

        prev_back = self.draws[-1][1]     # 上期后区
        skip_back = self.draws[-2][1]     # 上上期后区

        prev_hot = is_hot_combo(prev_back)
        skip_hot = is_hot_combo(skip_back)

        # 计算过热程度
        # 级别1：仅上期热（轻度过热）
        # 级别2：连续2期热（中度过热→启动冷三联）
        # 级别3：连续3期+热（重度过热→强力偏移）
        consecutive_hot = 0
        if prev_hot:
            consecutive_hot = 1
            if skip_hot:
                consecutive_hot = 2
                # 检查更早一期是否也热
                if len(self.draws) >= 4:
                    third_back = self.draws[-3][1]
                    if is_hot_combo(third_back):
                        consecutive_hot = 3

        if consecutive_hot < 2:
            # 未达到过热阈值，不启动冷三联
            return cold_force

        # 启动冷三联因子
        # 评分数值：冷号池中的号码得分，热号池中的号码降分
        cascade_strength = 0.30 if consecutive_hot == 2 else 0.45

        # 对每个后区号码计算冷三联评分
        for n in range(1, 13):
            in_cold = n in self.cold_pool
            in_hot = n in self.hot_pool

            if in_cold:
                # 冷号获得大力加分（冷三联核心：冷号回补）
                if consecutive_hot >= 3:
                    cold_force[n] = 0.90
                else:
                    cold_force[n] = 0.75
            elif not in_hot:
                # 中性号（既不在热池也不在冷池）：获得中等加分
                if consecutive_hot >= 3:
                    cold_force[n] = 0.60
                else:
                    cold_force[n] = 0.45
            else:
                # 热号降分（过热状态下热号不再继续热）
                cold_force[n] = 0.15

        print(f"[Back-Fusion] ❄️ 冷三联因子: 后区过热{consecutive_hot}期, "
              f"强度={cascade_strength:.2f}, "
              f"冷号奖励={[n for n in range(1,13) if cold_force[n]>=0.7]}")

        return cold_force

    def get_pool_summary(self) -> Dict[str, Any]:
        """
        获取当前池状态的摘要信息
        
        Returns:
            Dict: 各池号码列表和统计信息
        """
        return {
            'hot_pool': sorted(self.hot_pool),
            'cold_pool': sorted(self.cold_pool),
            'balance_pool': sorted(self.balance_pool),
            'hot_window': BACK_HOT_WINDOW,
            'cold_window': BACK_COLD_WINDOW,
            'hot_threshold': BACK_HOT_THRESHOLD,
            'cold_threshold': BACK_COLD_THRESHOLD,
            'strategy': self.BACK_STRATEGY,
            'data_periods': self.n,
        }

    def explain_recommendation(self, combo: List[int]) -> str:
        """
        解释推荐号码对的选号理由
        
        Args:
            combo: 后区号码对
        
        Returns:
            str: 详细解释文本
        """
        scores = self.score_back_combo(combo)
        n1, n2 = combo[0], combo[1]

        parts = []
        parts.append(f"【后区推荐】{n1}, {n2}")

        # 大小分析
        small = [n for n in combo if n in BACK_SIZE['small']]
        large = [n for n in combo if n in BACK_SIZE['large']]
        size_tag = "✅ 1:1理想" if len(small) == 1 else "⚡ 允许浮动"
        parts.append(f"大小比: {len(small)}:{len(large)} ({size_tag})")

        # 池入选
        hot_nums = [n for n in combo if n in self.hot_pool]
        cold_nums = [n for n in combo if n in self.cold_pool]
        balance_nums = [n for n in combo if n in self.balance_pool]
        
        if hot_nums:
            parts.append(f"🔥 含热号: {hot_nums}")
        if cold_nums:
            parts.append(f"❄️ 含冷号: {cold_nums}")
        if balance_nums:
            parts.append(f"⚖️ 含均衡号: {balance_nums}")

        # 综合评分
        parts.append(f"融合评分: {scores['fusion_score']:.4f}")
        parts.append(f"热号评分: {scores['hot_score']:.4f} | 冷号评分: {scores['cold_score']:.4f}")
        parts.append(f"均衡评分: {scores['balance_score']:.4f} | 博弈评分: {scores['game_score']:.4f}")
        parts.append(f"大小评分: {scores['size_score']:.4f}")

        return '\n'.join(parts)


# ============================================================
# 便捷函数
# ============================================================

def create_back_fusion(draws: List[Tuple[List[int], List[int]]]) -> BackZoneFusion:
    """
    创建后区融合器的便捷工厂函数
    
    Args:
        draws: 历史开奖数据
    
    Returns:
        BackZoneFusion: 初始化好的后区融合器
    """
    return BackZoneFusion(draws)


if __name__ == '__main__':
    # 简单测试
    print("🧪 后区融合模块测试")
    print("-" * 50)

    # 模拟历史数据（最近30期）
    test_draws: List[Tuple[List[int], List[int]]] = [
        ([5, 12, 18, 25, 33], [3, 8]),
        ([2, 9, 15, 24, 31], [1, 7]),
        ([7, 14, 20, 28, 34], [4, 11]),
        ([3, 11, 19, 26, 35], [2, 9]),
        ([8, 16, 22, 29, 32], [5, 10]),
        ([6, 10, 17, 23, 30], [3, 8]),
        ([1, 15, 20, 25, 34], [1, 5]),
        ([9, 14, 18, 28, 31], [7, 11]),
        ([5, 11, 16, 24, 35], [2, 9]),
        ([3, 8, 19, 27, 32], [4, 10]),
        ([7, 12, 22, 29, 33], [3, 12]),
        ([2, 10, 17, 26, 30], [1, 8]),
        ([6, 13, 20, 28, 34], [5, 11]),
        ([4, 9, 15, 23, 31], [2, 7]),
        ([8, 16, 21, 25, 35], [6, 10]),
        ([1, 11, 18, 27, 32], [3, 9]),
        ([5, 14, 19, 29, 33], [4, 12]),
        ([3, 10, 22, 24, 30], [1, 7]),
        ([7, 13, 17, 26, 34], [5, 8]),
        ([2, 9, 20, 28, 35], [2, 11]),
        ([6, 12, 16, 23, 31], [3, 10]),
        ([4, 15, 21, 27, 33], [1, 9]),
        ([8, 11, 18, 25, 32], [6, 12]),
        ([1, 14, 19, 29, 34], [4, 8]),
        ([5, 10, 17, 24, 30], [2, 7]),
        ([3, 13, 22, 26, 35], [5, 11]),
        ([7, 9, 16, 28, 31], [3, 10]),
        ([2, 12, 20, 27, 33], [1, 9]),
        ([6, 15, 18, 23, 34], [4, 12]),
        ([4, 7, 13, 21, 30], [2, 8]),
    ]

    # 初始化融合器
    fusion = BackZoneFusion(test_draws)

    # 打印池状态
    summary = fusion.get_pool_summary()
    print(f"热号池: {summary['hot_pool']}")
    print(f"冷号池: {summary['cold_pool']}")
    print(f"均衡池: {summary['balance_pool']}")
    print(f"策略权重: {summary['strategy']}")
    print()

    # 生成推荐
    recommendations = fusion.generate_recommendations(n=5)
    print(f"生成 {len(recommendations)} 个后区推荐:")
    for i, (combo, scores) in enumerate(recommendations, 1):
        size_info = f"{sum(1 for n in combo if n<=6)}:{sum(1 for n in combo if n>=7)}"
        print(f"\n#{i}: {combo[0]}, {combo[1]}  大小比={size_info}")
        print(f"   融合评分={scores['fusion_score']:.4f} final={scores.get('final_score', 'N/A')}")
        print(f"   热={scores['hot_score']:.3f} 冷={scores['cold_score']:.3f} 均衡={scores['balance_score']:.3f}")
        print(f"   博弈={scores['game_score']:.3f} 含热号={scores['in_hot']} 含冷号={scores['in_cold']}")

    print("\n" + "-" * 50)
    print("测试完成 ✅")

# Module-level alias for backward compatibility
BACK_STRATEGY = BackZoneFusion.BACK_STRATEGY
