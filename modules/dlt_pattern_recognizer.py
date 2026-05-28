#!/usr/bin/env python3
"""
DLT跨期模式识别器 (Pattern Recognizer)
=========================================
方案二核心实现：在五池体系上叠加跨期模式识别层。

设计理念：
- 彩票号码虽然随机，但号码组合的"模式"（如跨度、连号、尾号分布等）
  在统计上呈现出稳定的分布特征。
- 本模块从历史数据中提取9种独立模式特征的频率分布，
  对候选组合计算"模式匹配度"得分。
- 模式得分作为额外信号层，与现有池系统的冷热/频率得分融合，
  辅助遗传算法筛选更合理的号码组合。

9种模式特征:
1. 跨度 (Span) — 最大号-最小号
2. 连号 (Consecutive) — 相邻号码差为1的对数
3. 和值 (Sum) — 5个前区号码之和
4. 奇偶比 (Odd/Even Ratio) — 奇偶比例
5. 重号数 (Repeat) — 与上期重复的号码个数
6. 尾号分布 (Last Digit) — 号码末位数字分布
7. 三区分布 (Zone) — 1-12, 13-24, 25-35三个区间
8. 间隔值 (Gap) — 各号码距最近一次出现的期数
9. 质数计数 (Prime) — 质数个数
"""

from typing import List, Dict, Tuple, Any, Optional, Set
from collections import Counter, defaultdict
import numpy as np
import math
import random


class DLTPatternRecognizer:
    """
    跨期模式识别器
    
    从历史开奖数据中提取模式分布，对候选组合进行模式匹配度评分。
    可作为第6个池（模式池）的生成器，也可作为现有池的评分增强层。
    """
    
    # 前区质数集合 (1-35)
    PRIME_NUMBERS = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
    # 前区三区划分
    ZONE1 = range(1, 13)    # 1-12
    ZONE2 = range(13, 25)    # 13-24
    ZONE3 = range(25, 36)    # 25-35

    def __init__(self, draws: List[Tuple[List[int], List[int]]]):
        """
        初始化模式识别器
        
        Args:
            draws: 历史开奖数据，[(前区5号, 后区2号), ...]
        """
        self.draws = draws
        self.n_periods = len(draws)
        
        # 模式频率分布缓存
        self._pattern_distributions: Dict[str, Dict] = {}
        self._is_built = False
        
    # ================================================================
    # 模式特征提取（静态方法，可独立使用）
    # ================================================================
    
    @staticmethod
    def extract_span(front: List[int]) -> int:
        """跨度：最大号-最小号"""
        return max(front) - min(front)
    
    @staticmethod
    def extract_consecutive_count(front: List[int]) -> int:
        """连号对数：相邻差为1的对数"""
        s = sorted(front)
        count = 0
        for i in range(len(s) - 1):
            if s[i+1] - s[i] == 1:
                count += 1
            elif s[i+1] - s[i] == 2:
                count += 0.5  # 隔号（如 9,11）作为半连号
        return count
    
    @staticmethod
    def extract_sum(front: List[int]) -> int:
        """和值"""
        return sum(front)
    
    @staticmethod
    def extract_odd_even_ratio(front: List[int]) -> Tuple[int, int]:
        """奇偶比 (奇个数, 偶个数)"""
        odd = sum(1 for n in front if n % 2 == 1)
        return (odd, 5 - odd)
    
    @staticmethod
    def extract_repeat_count(front: List[int], prev_front: List[int]) -> int:
        """与上期重复的号码个数"""
        return len(set(front) & set(prev_front))
    
    @staticmethod
    def extract_last_digits(front: List[int]) -> List[int]:
        """尾号：各号码个位数"""
        return [n % 10 for n in front]
    
    @staticmethod
    def extract_zone_distribution(front: List[int]) -> Tuple[int, int, int]:
        """三区分布：(1区个数, 2区个数, 3区个数)"""
        z1 = sum(1 for n in front if n in DLTPatternRecognizer.ZONE1)
        z2 = sum(1 for n in front if n in DLTPatternRecognizer.ZONE2)
        z3 = sum(1 for n in front if n in DLTPatternRecognizer.ZONE3)
        return (z1, z2, z3)
    
    @staticmethod
    def extract_zone_pattern(front: List[int]) -> str:
        """三区分布模式字符串，如 '122'"""
        z1, z2, z3 = DLTPatternRecognizer.extract_zone_distribution(front)
        return f"{z1}{z2}{z3}"
    
    @staticmethod
    def extract_gaps(front: List[int], history: List[List[int]],
                     back: Optional[List[int]] = None,
                     back_history: Optional[List[List[int]]] = None) -> Dict[int, int]:
        """
        各号码距最近一次出现的期数间隔
        
        Args:
            front: 当前前区号码组合
            history: 前区历史号码列表（每期5个号码的列表）
            back: 当前后区号码组合（可选）
            back_history: 后区历史号码列表（可选）
            
        Returns:
            Dict: {号码: 间隔期数}
        """
        gaps = {}
        history_rev = list(reversed(history))
        for n in front:
            gap = 1
            found = False
            for period_nums in history_rev:
                if n in period_nums:
                    gaps[n] = gap
                    found = True
                    break
                gap += 1
            if not found:
                gaps[n] = len(history)  # 从未出现
        
        if back is not None and back_history is not None:
            back_history_rev = list(reversed(back_history))
            for n in back:
                gap = 1
                found = False
                for period_nums in back_history_rev:
                    if n in period_nums:
                        gaps[n] = gap
                        found = True
                        break
                    gap += 1
                if not found:
                    gaps[n] = len(back_history)
        
        return gaps
    
    @staticmethod
    def extract_prime_count(front: List[int]) -> int:
        """质数个数"""
        return sum(1 for n in front if n in DLTPatternRecognizer.PRIME_NUMBERS)
    
    @staticmethod
    def extract_ac_value(front: List[int]) -> float:
        """
        AC值（算术复杂度）: 衡量号码组合的离散程度。
        
        计算方式：所有两两差的绝对值中，不同值的个数，减去 (号码个数-1)
        前区5个号码，AC值范围通常在1-10之间。
        高频AC值区间为4-8（覆盖约80%）。
        """
        s = sorted(front)
        diffs = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diffs.add(abs(s[j] - s[i]))
        return len(diffs) - (len(s) - 1)
    
    @staticmethod
    def extract_all_front_patterns(front: List[int],
                                    prev_front: Optional[List[int]] = None,
                                    history: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """
        提取单个前区组合的所有模式特征
        
        Returns:
            Dict: 包含所有模式特征
        """
        features = {
            'span': DLTPatternRecognizer.extract_span(front),
            'consecutive': DLTPatternRecognizer.extract_consecutive_count(front),
            'sum': DLTPatternRecognizer.extract_sum(front),
            'odd_even': DLTPatternRecognizer.extract_odd_even_ratio(front),
            'last_digits': tuple(sorted(DLTPatternRecognizer.extract_last_digits(front))),
            'zone_pattern': DLTPatternRecognizer.extract_zone_pattern(front),
            'zone_dist': DLTPatternRecognizer.extract_zone_distribution(front),
            'prime_count': DLTPatternRecognizer.extract_prime_count(front),
            'ac_value': DLTPatternRecognizer.extract_ac_value(front),
        }
        if prev_front is not None:
            features['repeat_count'] = DLTPatternRecognizer.extract_repeat_count(front, prev_front)
        if history is not None:
            gaps = DLTPatternRecognizer.extract_gaps(front, history)
            features['avg_gap'] = float(np.mean(list(gaps.values()))) if gaps else 0
            features['min_gap'] = min(gaps.values()) if gaps else 0
            features['max_gap'] = max(gaps.values()) if gaps else 0
        
        return features
    
    # ================================================================
    # 模式分布构建
    # ================================================================
    
    def build_distributions(self, window: int = 500) -> None:
        """
        从历史数据构建各模式的频率分布表
        
        Args:
            window: 使用的历史期数窗口（默认最近500期，兼顾统计稳定性和时效性）
        """
        if len(self.draws) < 50:
            print("[Pattern-Recognizer] ⚠️ 历史数据不足50期，模式分析可能不准确")
        
        # 使用最近window期
        history = self.draws[-window:] if len(self.draws) > window else self.draws
        front_history = [d[0] for d in history]
        
        # 1. 跨度分布
        span_counter = Counter()
        for f in front_history:
            span_counter[self.extract_span(f)] += 1
        
        # 2. 连号分布
        consecutive_counter = Counter()
        for f in front_history:
            cc = self.extract_consecutive_count(f)
            consecutive_counter[cc] += 1
        
        # 3. 和值分布（按5分组）
        sum_counter = Counter()
        for f in front_history:
            s = self.extract_sum(f)
            bucket = (s // 5) * 5  # 5为组距
            sum_counter[bucket] += 1
        
        # 4. 奇偶比分布
        oe_counter = Counter()
        for f in front_history:
            oe_counter[self.extract_odd_even_ratio(f)] += 1
        
        # 5. 重号分布（需要相邻两期）
        repeat_counter = Counter()
        for i in range(1, len(front_history)):
            cnt = self.extract_repeat_count(front_history[i], front_history[i-1])
            repeat_counter[cnt] += 1
        
        # 6. 尾号分布（聚合尾号频率）
        last_digit_counter = Counter()
        for f in front_history:
            for d in self.extract_last_digits(f):
                last_digit_counter[d] += 1
        
        # 7. 三区模式分布
        zone_pattern_counter = Counter()
        for f in front_history:
            zone_pattern_counter[self.extract_zone_pattern(f)] += 1
        
        # 8. 质数个数分布
        prime_counter = Counter()
        for f in front_history:
            prime_counter[self.extract_prime_count(f)] += 1
        
        # 9. AC值分布
        ac_counter = Counter()
        for f in front_history:
            ac_counter[self.extract_ac_value(f)] += 1
        
        total = len(front_history)
        
        self._pattern_distributions = {
            'span': {
                'counter': span_counter,
                'total': total,
                'type': 'categorical',
                'weight': 0.12,
                'desc': '跨度'
            },
            'consecutive': {
                'counter': consecutive_counter,
                'total': total,
                'type': 'categorical',
                'weight': 0.20,
                'desc': '连号'
            },
            'sum': {
                'counter': sum_counter,
                'total': total,
                'type': 'bucketed',
                'weight': 0.12,
                'desc': '和值'
            },
            'odd_even': {
                'counter': oe_counter,
                'total': total,
                'type': 'categorical',
                'weight': 0.10,
                'desc': '奇偶比'
            },
            'repeat': {
                'counter': repeat_counter,
                'total': total - 1 if total > 1 else 1,
                'type': 'categorical',
                'weight': 0.15,
                'desc': '重号'
            },
            'last_digit': {
                'counter': last_digit_counter,
                'total': total * 5,
                'type': 'frequency',
                'weight': 0.10,
                'desc': '尾号'
            },
            'zone_pattern': {
                'counter': zone_pattern_counter,
                'total': total,
                'type': 'categorical',
                'weight': 0.12,
                'desc': '三区分布'
            },
            'prime': {
                'counter': prime_counter,
                'total': total,
                'type': 'categorical',
                'weight': 0.07,
                'desc': '质数'
            },
            'ac_value': {
                'counter': ac_counter,
                'total': total,
                'type': 'categorical',
                'weight': 0.12,
                'desc': 'AC值'
            },
        }
        
        self._is_built = True
        # print(f"[Pattern-Recognizer] ✅ 模式分布构建完成 ({total}期)")
    
    # ================================================================
    # 模式匹配评分
    # ================================================================
    
    def score_combo(self, front: List[int],
                    prev_front: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        对单个前区组合进行模式匹配评分
        
        每个模式的得分 = 该模式值在历史中的频率百分比 × 权重
        
        Args:
            front: 前区号码组合（5个号码）
            prev_front: 上期前区（用于重号计算）
            
        Returns:
            Dict: {
                'total_score': float,      # 综合模式匹配度 (0-1)
                'detail': Dict[str, float] # 各模式得分
                'patterns': Dict           # 各模式的具体值
            }
        """
        if not self._is_built:
            self.build_distributions()
        
        # 提取特征
        features = self.extract_all_front_patterns(
            front, prev_front=prev_front
        )
        
        detail = {}
        total_weight = 0
        weighted_sum = 0
        
        # 1. 跨度匹配度
        dist = self._pattern_distributions['span']
        span_prob = dist['counter'].get(features['span'], 0) / dist['total']
        detail['span'] = span_prob * dist['weight']
        weighted_sum += span_prob * dist['weight']
        total_weight += dist['weight']
        
        # 2. 连号匹配度
        dist = self._pattern_distributions['consecutive']
        cc_prob = dist['counter'].get(features['consecutive'], 0) / dist['total']
        detail['consecutive'] = cc_prob * dist['weight']
        weighted_sum += cc_prob * dist['weight']
        total_weight += dist['weight']
        
        # 3. 和值匹配度
        dist = self._pattern_distributions['sum']
        sum_bucket = (features['sum'] // 5) * 5
        sum_prob = dist['counter'].get(sum_bucket, 0) / dist['total']
        detail['sum'] = sum_prob * dist['weight']
        weighted_sum += sum_prob * dist['weight']
        total_weight += dist['weight']
        
        # 4. 奇偶比匹配度
        dist = self._pattern_distributions['odd_even']
        oe_prob = dist['counter'].get(features['odd_even'], 0) / dist['total']
        detail['odd_even'] = oe_prob * dist['weight']
        weighted_sum += oe_prob * dist['weight']
        total_weight += dist['weight']
        
        # 5. 重号匹配度
        dist = self._pattern_distributions['repeat']
        if prev_front is not None and 'repeat_count' in features:
            rp_prob = dist['counter'].get(features['repeat_count'], 0) / dist['total']
        else:
            rp_prob = 0.5  # 未提供上期数据，中性评分
        detail['repeat'] = rp_prob * dist['weight']
        weighted_sum += rp_prob * dist['weight']
        total_weight += dist['weight']
        
        # 6. 尾号匹配度（组合的尾号多样性）
        dist = self._pattern_distributions['last_digit']
        digits = features['last_digits']
        digit_scores = []
        for d in digits:
            digit_scores.append(dist['counter'].get(d, 0) / max(dist['total'], 1))
        digit_score = float(np.mean(digit_scores)) if digit_scores else 0.5
        
        # 同时奖励尾号多样性（5个号中不同尾号越多越好）
        unique_digits = len(set(digits))
        diversity_bonus = min(unique_digits / 5.0, 1.0) * 0.3
        digit_score = digit_score * 0.7 + diversity_bonus * 0.3
        
        detail['last_digit'] = digit_score * dist['weight']
        weighted_sum += digit_score * dist['weight']
        total_weight += dist['weight']
        
        # 7. 三区模式匹配度
        dist = self._pattern_distributions['zone_pattern']
        zp_prob = dist['counter'].get(features['zone_pattern'], 0) / dist['total']
        detail['zone_pattern'] = zp_prob * dist['weight']
        weighted_sum += zp_prob * dist['weight']
        total_weight += dist['weight']
        
        # 8. 质数个数匹配度
        dist = self._pattern_distributions['prime']
        prime_prob = dist['counter'].get(features['prime_count'], 0) / dist['total']
        detail['prime'] = prime_prob * dist['weight']
        weighted_sum += prime_prob * dist['weight']
        total_weight += dist['weight']
        
        # 9. AC值匹配度
        dist = self._pattern_distributions['ac_value']
        ac_prob = dist['counter'].get(features['ac_value'], 0) / dist['total']
        detail['ac_value'] = ac_prob * dist['weight']
        weighted_sum += ac_prob * dist['weight']
        total_weight += dist['weight']
        
        total_score = weighted_sum / max(total_weight, 0.01)
        
        return {
            'total_score': round(min(total_score, 1.0), 4),
            'detail': detail,
            'patterns': features,
        }
    
    def score_candidates(self, candidates: List[List[int]],
                         prev_front: Optional[List[int]] = None,
                         back_candidates: Optional[List[List[int]]] = None) -> List[Dict[str, Any]]:
        """
        对候选组合列表批量评分
        
        Args:
            candidates: 候选前区组合列表
            prev_front: 上期前区（用于重号计算）
            back_candidates: 后区候选（可选，将来扩展后区模式）
            
        Returns:
            List[Dict]: 每个候选的评分结果
        """
        if not self._is_built:
            self.build_distributions()
        
        results = []
        for front in candidates:
            score = self.score_combo(front, prev_front)
            results.append({
                'front': front,
                'pattern_score': score['total_score'],
                'details': score['detail'],
                'patterns': score['patterns'],
            })
        
        return results
    
    # ================================================================
    # 模式池生成（第6池 - 模式推荐池）
    # ================================================================
    
    def generate_pattern_pool(self, n_front: int = 10, n_back: int = 4,
                               top_patterns: int = 3) -> Tuple[List[int], List[int]]:
        """
        模式池：基于历史高频模式特征，生成推荐的号码集合。
        
        策略：对每种模式，取该模式下在历史中出现频率最高的值，
        然后筛选符合条件的号码汇总。
        
        Args:
            n_front: 返回的前区推荐数量
            n_back: 返回的后区推荐数量
            top_patterns: 每种模式取Top N高频值
            
        Returns:
            (前区推荐列表, 后区推荐列表)
        """
        if not self._is_built:
            self.build_distributions()
        
        # ===== 前区模式池 =====
        # 从各模式的高频值推导推荐号码
        recommended_numbers = set()
        
        # 策略1: 从高频三区模式反推号码
        zone_dist = self._pattern_distributions['zone_pattern']
        top_zone_patterns = [zp for zp, _ in zone_dist['counter'].most_common(top_patterns)]
        
        for zp in top_zone_patterns:
            z1, z2, z3 = int(zp[0]), int(zp[1]), int(zp[2])
            # 从各区选号时，融合该区的冷热信息
            zone_numbers = self._pick_numbers_by_zone(z1, z2, z3)
            recommended_numbers.update(zone_numbers)
        
        # 策略2: 从高频跨度反推（跨度决定了最大最小号的差值）
        span_dist = self._pattern_distributions['span']
        top_spans = [s for s, _ in span_dist['counter'].most_common(top_patterns)]
        
        # 从跨度和和值的交集反推号码组合
        sum_dist = self._pattern_distributions['sum']
        top_sums = [s for s, _ in sum_dist['counter'].most_common(top_patterns)]
        
        # 对每个跨度+和值组合，生成推荐号
        for span in top_spans:
            for sum_bucket in top_sums[:2]:
                # 在号码范围内找满足约束的组合：和值≈sum_bucket±5, 跨度≈span±2
                candidates = self._find_numbers_by_span_sum(span, sum_bucket)
                recommended_numbers.update(candidates)
        
        # 策略3: 从高频质数个数的组合中选号
        prime_dist = self._pattern_distributions['prime']
        top_prime_counts = [p for p, _ in prime_dist['counter'].most_common(2)]
        
        for pc in top_prime_counts:
            if pc > 0:
                # 选质数 + 非质数混合
                primes = sorted(self.PRIME_NUMBERS)
                non_primes = [n for n in range(1, 36) if n not in self.PRIME_NUMBERS]
                
                # 加上近期热门质数
                recent_primes = self._get_recent_numbers(prime_only=True, n=5)
                recommended_numbers.update(recent_primes)
                
                # 加上高频非质数
                recent_non_primes = self._get_recent_numbers(prime_only=False, n=5)
                recommended_numbers.update(recent_non_primes)
        
        # 分区均衡截取：确保各区按比例覆盖
        # 先按号码范围排序后，按区分配名额
        sorted_nums = sorted(recommended_numbers)
        z1 = [n for n in sorted_nums if n in self.ZONE1]
        z2 = [n for n in sorted_nums if n in self.ZONE2]
        z3 = [n for n in sorted_nums if n in self.ZONE3]
        
        # 按 ≈1:2:2 比例分配（与常见的122/212/221三区分布一致）
        zone1_quota = max(2, n_front // 5)
        zone2_quota = max(3, n_front * 2 // 5)
        zone3_quota = max(3, n_front * 2 // 5)
        
        selected = []
        selected.extend(z1[:zone1_quota])
        selected.extend(z2[:zone2_quota])
        selected.extend(z3[:zone3_quota])
        
        front_pool = selected[:n_front]
        
        # 不足时用频率补充
        if len(front_pool) < n_front:
            freq_sorted = self._get_frequency_sorted('front')
            for n in freq_sorted:
                if n not in front_pool:
                    front_pool.append(n)
                    if len(front_pool) >= n_front:
                        break
        
        # ===== 后区模式池 =====
        back_pool = self._generate_back_pattern_pool(n_back)
        
        print(f"🧩 生成前区模式池: {front_pool}")
        print(f"🧩 生成后区模式池: {back_pool}")
        
        return front_pool, back_pool
    
    def _pick_numbers_by_zone(self, z1: int, z2: int, z3: int) -> List[int]:
        """根据三区分布推荐号码，各区按出现频率排序后取前几"""
        front_history = [d[0] for d in self.draws[-100:]]
        
        zone_freq = {1: Counter(), 2: Counter(), 3: Counter()}
        for f in front_history:
            for n in f:
                if n in self.ZONE1:
                    zone_freq[1][n] += 1
                elif n in self.ZONE2:
                    zone_freq[2][n] += 1
                else:
                    zone_freq[3][n] += 1
        
        result = []
        zone_nums = [z1, z2, z3]
        zone_ranges = [self.ZONE1, self.ZONE2, self.ZONE3]
        
        for zi in range(3):
            needed = zone_nums[zi]
            if needed <= 0:
                continue
            # 取该区频率最高的 needed*3 个号码
            top_nums = [n for n, _ in zone_freq[zi+1].most_common(needed * 3)]
            if not top_nums:
                # 该区无历史数据，随机补充
                top_nums = random.sample(list(zone_ranges[zi]), min(needed * 3, len(zone_ranges[zi])))
            result.extend(top_nums)
        
        return result
    
    def _find_numbers_by_span_sum(self, target_span: int, target_sum_bucket: int) -> List[int]:
        """
        根据跨度+和值约束推荐号码。
        从历史中找满足近似约束的号码组合，返回推荐号码集合。
        """
        front_history = [d[0] for d in self.draws[-200:]]
        candidates = set()
        
        for f in front_history:
            span = self.extract_span(f)
            s = self.extract_sum(f)
            s_bucket = (s // 5) * 5
            if abs(span - target_span) <= 3 and s_bucket == target_sum_bucket:
                candidates.update(f)
                if len(candidates) > 20:
                    break
        
        # 如果候选不足，放宽条件
        if len(candidates) < 5:
            for f in front_history:
                span = self.extract_span(f)
                if abs(span - target_span) <= 5:
                    candidates.update(f)
                    if len(candidates) > 15:
                        break
        
        return list(candidates)
    
    def _get_recent_numbers(self, prime_only: bool = True, n: int = 5) -> List[int]:
        """获取近期频繁出现的质数/非质数"""
        front_history = [d[0] for d in self.draws[-50:]]
        counter = Counter()
        for f in front_history:
            for num in f:
                is_prime = num in self.PRIME_NUMBERS
                if (prime_only and is_prime) or (not prime_only and not is_prime):
                    counter[num] += 1
        return [num for num, _ in counter.most_common(n)]
    
    def _get_frequency_sorted(self, zone: str) -> List[int]:
        """按出现频率排序的号码"""
        if zone == 'front':
            balls = [d[0] for d in self.draws[-100:]]
            all_nums = range(1, 36)
        else:
            balls = [d[1] for d in self.draws[-100:]]
            all_nums = range(1, 13)
        
        counter = Counter()
        for draw in balls:
            for n in draw:
                counter[n] += 1
        
        return [n for n, _ in counter.most_common()]
    
    def _generate_back_pattern_pool(self, n: int = 4) -> List[int]:
        """生成后区模式池"""
        back_history = [d[1] for d in self.draws[-100:]]
        
        # 后区模式特征：高频+近期热门+奇偶均衡
        back_counter = Counter()
        for b in back_history:
            for n in b:
                back_counter[n] += 1
        
        # 近期热门
        recent_backs = back_history[-30:] if len(back_history) > 30 else back_history
        recent_counter = Counter()
        for b in recent_backs:
            for n in b:
                recent_counter[n] += 1
        
        # 遗漏值：距上次出现的期数
        last_occurrence = {}
        for i, b in enumerate(reversed(back_history)):
            for n in b:
                if n not in last_occurrence:
                    last_occurrence[n] = i + 1
        for n in range(1, 13):
            if n not in last_occurrence:
                last_occurrence[n] = len(back_history)
        
        # 综合评分 = 近期频率×0.5 + 总频率×0.3 + 小量随机扰动防平局
        max_total = max(back_counter.values()) if back_counter else 1
        max_recent = max(recent_counter.values()) if recent_counter else 1
        
        scores = {}
        for n in range(1, 13):
            total_score = back_counter.get(n, 0) / max_total if max_total > 0 else 0
            recent_score = recent_counter.get(n, 0) / max_recent if max_recent > 0 else 0
            # 加入极小随机扰动避免平局（seed基于号码固定，保持可复现）
            tiebreaker = (n * 0.0001) % 1
            scores[n] = total_score * 0.3 + recent_score * 0.5 + last_occurrence.get(n, 0) / max(len(back_history), 1) * 0.15 + tiebreaker * 0.05
        
        sorted_backs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_backs[:n]
    
    # ================================================================
    # 分析报告
    # ================================================================
    
    def generate_pattern_report(self, top_n: int = 5) -> str:
        """生成模式分布分析报告"""
        if not self._is_built:
            self.build_distributions()
        
        lines = []
        lines.append("=" * 55)
        lines.append("  📊 跨期模式频率分布报告")
        lines.append(f"  分析期数: {self.n_periods}")
        lines.append("=" * 55)
        
        for key in ['span', 'consecutive', 'sum', 'odd_even', 'repeat',
                     'zone_pattern', 'prime', 'ac_value']:
            dist = self._pattern_distributions[key]
            lines.append(f"\n  [{dist['desc']}] (权重:{dist['weight']})")
            lines.append(f"  {'-' * 40}")
            for val, cnt in dist['counter'].most_common(top_n):
                pct = cnt / dist['total'] * 100
                bar = '█' * int(pct / 2) + '░' * max(0, 20 - int(pct / 2))
                lines.append(f"    {str(val):>6s}: {pct:5.1f}% {bar}")
        
        lines.append(f"\n{'=' * 55}")
        return "\n".join(lines)
    
    def analyze_combo_pattern_fit(self, combo: List[int]) -> Dict[str, Any]:
        """
        分析单个组合与各高频模式的匹配情况。
        用于调试和可视化。
        """
        features = self.extract_all_front_patterns(combo)
        score_result = self.score_combo(combo)
        
        analysis = {
            'combo': combo,
            'overall_pattern_score': score_result['total_score'],
            'features': features,
            'pattern_fit': {},
        }
        
        for key in ['span', 'consecutive', 'sum', 'odd_even',
                     'zone_pattern', 'prime', 'ac_value']:
            dist = self._pattern_distributions[key]
            feature_key = key
            if key == 'odd_even':
                val = features.get('odd_even', (0, 0))
            elif key == 'zone_pattern':
                val = features.get('zone_pattern', '000')
            else:
                val = features.get(key, 0)
            
            # 模式值在当前分布的百分位排名
            total = dist['total']
            rank = sum(v for k, v in dist['counter'].items()
                       if isinstance(k, type(val)) and k <= val) / max(total, 1) * 100
            
            analysis['pattern_fit'][key] = {
                'value': val,
                'frequency_pct': round(dist['counter'].get(val, 0) / max(total, 1) * 100, 1),
                'percentile_rank': round(rank, 1),
            }
        
        return analysis
    
    def generate_pattern_pool_compound(self, n_front: int = 7,
                                        n_back: int = 3) -> Dict[str, Any]:
        """
        生成基于模式池的复式投注推荐。
        
        Returns:
            Dict: { 'name': '模式池', 'front': [...], 'back': [...], 'strategy': '...' }
        """
        front_pool, back_pool = self.generate_pattern_pool(
            n_front=n_front, n_back=n_back
        )
        
        return {
            'name': f'{n_front}+{n_back}复式(模式池)',
            'front': sorted(front_pool),
            'back': sorted(back_pool),
            'strategy': '模式池: 基于9种跨期模式特征的匹配度推荐',
            'description': '提取历史高频模式特征，生成与历史中奖模式最接近的号码组合',
        }


# ================================================================
# 便捷函数：模式评分与现有池系统融合
# ================================================================

def apply_pattern_boost(candidates: List[Dict[str, Any]],
                         pattern_recognizer: DLTPatternRecognizer,
                         prev_front: Optional[List[int]] = None,
                         boost_weight: float = 0.25) -> List[Dict[str, Any]]:
    """
    对候选列表应用模式评分增强。
    
    在现有 base_score + gt_score + genetic_score 基础上，
    叠加 pattern_score，调整 final_score。
    
    Args:
        candidates: 候选列表（含 front, base_score, final_score 等字段）
        pattern_recognizer: 模式识别器实例
        prev_front: 上期前区号码
        boost_weight: 模式评分的权重
        
    Returns:
        更新 final_score 后的候选列表
    """
    for c in candidates:
        front = c.get('front', [])
        if not front:
            continue
        
        score_result = pattern_recognizer.score_combo(front, prev_front)
        pattern_score = score_result['total_score']
        
        # 记录原比分
        orig_final = c.get('final_score', c.get('base_score', 0.5))
        
        # 新最终分 = 原分 × (1 - boost_weight) + pattern_score × boost_weight
        new_final = orig_final * (1 - boost_weight) + pattern_score * boost_weight
        
        c['original_final_score'] = orig_final
        c['pattern_score'] = pattern_score
        c['final_score'] = new_final
    
    return candidates


def generate_pattern_diversity_pool(draws: List[Tuple[List[int], List[int]]],
                                     n_front: int = 12, n_back: int = 4,
                                     top_patterns: int = 5) -> Tuple[List[int], List[int]]:
    """
    基于模式多样性生成候选池。
    
    与模式池不同：模式池取高频模式值，多样性池取不同模式值的组合，
    确保覆盖多样化的模式。
    
    Args:
        draws: 历史开奖数据
        n_front: 前区推荐数
        n_back: 后区推荐数
        top_patterns: 每种模式考虑的类别数
        
    Returns:
        (前区号码, 后区号码)
    """
    recognizer = DLTPatternRecognizer(draws)
    recognizer.build_distributions()
    
    front_history = [d[0] for d in draws[-200:]]
    
    # 对每种模式，选择高频模式值下的代表性号码
    zone_pattern_counter = Counter()
    span_counter = Counter()
    sum_counter = Counter()
    ac_counter = Counter()
    
    for f in front_history:
        zone_pattern_counter[recognizer.extract_zone_pattern(f)] += 1
        span_counter[recognizer.extract_span(f)] += 1
        s = recognizer.extract_sum(f)
        sum_counter[(s // 5) * 5] += 1
        ac_counter[recognizer.extract_ac_value(f)] += 1
    
    # 收集每个模式Top类别下的号码
    diversity_numbers = set()
    
    for zp, _ in zone_pattern_counter.most_common(top_patterns):
        z1, z2, z3 = int(zp[0]), int(zp[1]), int(zp[2])
        nums = recognizer._pick_numbers_by_zone(z1, z2, z3)
        diversity_numbers.update(nums[:6])
    
    # 补充近期出现在高频AC值区间内的号码
    for ac_val, _ in ac_counter.most_common(3):
        for f in front_history:
            if recognizer.extract_ac_value(f) == ac_val:
                diversity_numbers.update(f)
                if len(diversity_numbers) > n_front * 2:
                    break
    
    front_result = sorted(diversity_numbers)[:n_front]
    
    # 不足时用频率补充
    if len(front_result) < n_front:
        freq_sorted = recognizer._get_frequency_sorted('front')
        for n in freq_sorted:
            if n not in front_result:
                front_result.append(n)
                if len(front_result) >= n_front:
                    break
    
    # 后区
    back_pool = recognizer._generate_back_pattern_pool(n_back)
    
    return front_result[:n_front], back_pool[:n_back]
