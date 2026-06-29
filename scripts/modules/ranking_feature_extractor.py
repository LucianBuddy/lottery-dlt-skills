#!/usr/bin/env python3
"""
DLT Ranking Feature Extractor V1.0

将候选号码转换为特征向量，用于替代20步串行评分的可学习排序模型。

特征分类（~55维）：
  [0-9]   位置频率特征
  [10-19] 冷热号特征
  [20-29] 区域分布特征
  [30-39] 数学结构特征
  [40-49] 跨期模式特征
  [50-54] 元特征(策略/池归属)
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter


# ============================================================
# 常量
# ============================================================

ZONE1 = set(range(1, 13))
ZONE2 = set(range(13, 25))
ZONE3 = set(range(25, 36))
PRIMES_FRONT = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
ALL_FRONT = list(range(1, 36))


def ac_value(numbers: List[int]) -> int:
    """计算AC值"""
    n = len(numbers)
    if n < 2:
        return 0
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(abs(numbers[i] - numbers[j]))
    return len(diffs) - (n - 1)


def compute_tail_density(numbers: List[int]) -> float:
    """尾号聚合度：不同尾号的数量 / 号码数量（越低越聚合）"""
    tails = set(n % 10 for n in numbers)
    return len(tails) / len(numbers)


def compute_scatter(numbers: List[int]) -> float:
    """散度：相邻号码间距的变异系数"""
    if len(numbers) < 2:
        return 0.0
    gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
    if not gaps or sum(gaps) == 0:
        return 0.0
    return float(np.std(gaps)) / float(np.mean(gaps) + 1e-8)


# ============================================================
# 【方向B】号码embedding (PCA降维自共现矩阵)
# ============================================================

def _build_number_embeddings(draws, zone='front', dim=2):
    """
    从号码共现矩阵构建2D embedding。
    共现矩阵 M[i][j] = 号码i和j在同一期出现的次数。
    PCA降维到dim维，作为号码的隐式表示。
    """
    try:
        from sklearn.decomposition import PCA
        import numpy as np

        n_range = 35 if zone == 'front' else 12
        # 共现矩阵
        cooc = np.zeros((n_range, n_range))
        for d in draws:
            nums = d[0] if zone == 'front' else d[1]
            for i in nums:
                for j in nums:
                    if i != j:
                        cooc[i-1][j-1] += 1

        # PCA到2D
        pca = PCA(n_components=dim, random_state=42)
        emb = pca.fit_transform(cooc)
        # 归一化到[-1, 1]
        emb = emb / (np.max(np.abs(emb)) + 1e-8)
        return {n+1: emb[n].tolist() for n in range(n_range)}
    except Exception:
        return {}


# ============================================================
# 交叉特征计算
# ============================================================

def _compute_cross_features(front, back, draws):
    """
    计算10个交叉特征，用于增强模型的非线性捕获能力。

    Returns: List[float] (10维)
    """
    n = len(draws)
    fsum = sum(front)
    fset = set(front)

    # 1. 频率×遗漏: 热号出现多但遗漏少 = 高确定性
    freq_omit = 0.0
    omit_prod = 0.0
    for num in front:
        freq = sum(1 for d in draws[-30:] if num in d[0]) / 30.0
        omit = 0
        for d in reversed(draws):
            if num in d[0]:
                break
            omit += 1
        freq_omit += freq * (omit / max(n, 1))
        omit_prod += (freq + 0.01) * (omit + 1)
    freq_omit /= 5.0
    omit_prod /= 5.0

    # 2. 和值×跨度
    span = max(front) - min(front)
    sum_span = (fsum / 150.0) * (span / 34.0)

    # 3. 奇偶×012路
    odd = sum(1 for n in front if n % 2 == 1) / 5.0
    road0 = sum(1 for n in front if n % 3 == 0) / 5.0
    road1 = sum(1 for n in front if n % 3 == 1) / 5.0
    parity_road = odd * road0

    # 4. Z1占比×Z3占比 (极端分布检测)
    z1 = len([n for n in front if n <= 12]) / 5.0
    z3 = len([n for n in front if n >= 25]) / 5.0
    zone_extreme = z1 * z3  # 同时覆盖两极时较高

    # 5. 重号率×冷号率
    if n >= 2:
        prev = set(draws[-1][0])
        repeat = len(fset & prev) / 5.0
    else:
        repeat = 0.0
    cold = sum(1 for n in front if n >= 25) / 5.0  # 大号作为冷号代理
    repeat_cold = repeat * cold

    # 6. AC值×跨度
    ac = ac_value(front) / 15.0
    ac_span = ac * (span / 34.0)

    # 7. 尾号密度×散度
    tails = set(n % 10 for n in front)
    tail_d = len(tails) / 5.0
    gaps = [front[i+1]-front[i] for i in range(len(front)-1)] if len(front) >= 2 else [1]
    scatter = float(np.std(gaps)) / max(float(np.mean(gaps)), 1) if hasattr(np, 'std') else 0.5
    tail_scatter = tail_d * min(scatter, 2.0) / 2.0

    # 8. 连号×和值偏度
    consec = sum(1 for i in range(len(front)-1) if front[i+1]-front[i] == 1)
    sum_skew = (fsum - 90) / 60.0  # 和值偏离标准范围的程度
    consec_sum = (consec / 4.0) * abs(sum_skew)

    # 9. 后区奇偶×前区奇偶
    back_odd = sum(1 for n in back if n % 2 == 1) / 2.0
    parity_cross = odd * back_odd

    # 10. 前区最大×后区最小
    front_max = max(front) / 35.0
    back_min = min(back) / 12.0
    max_min_cross = front_max * back_min

    return [
        min(freq_omit, 1.0),
        min(omit_prod / 10.0, 1.0),
        min(sum_span, 1.0),
        min(parity_road, 1.0),
        min(zone_extreme, 1.0),
        min(repeat_cold * 2.0, 1.0),
        min(ac_span * 2.0, 1.0),
        min(tail_scatter, 1.0),
        min(consec_sum, 1.0),
        min(parity_cross, 1.0),
        min(max_min_cross, 1.0),
    ]


import numpy as np


def extract_features(
    candidate: Dict[str, Any],
    draws: List[Tuple[List[int], List[int]]],
    draw_periods: Optional[List[int]] = None,
    pattern_scores: Optional[Dict[str, float]] = None,
) -> List[float]:
    """
    从候选号码中提取特征向量。

    Args:
        candidate: {'front': [...], 'back': [...], 'base_score': ..., 'gt_score': ..., ...}
        draws: 完整历史开奖数据
        draw_periods: 期号列表（与draws等长）
        pattern_scores: 跨期模式评分缓存（可选）

    Returns:
        List[float]: 特征向量
    """
    front = sorted(candidate.get('front', []))
    back = candidate.get('back', [1, 12])
    n_draws = len(draws)

    features = []

    # ============================================================
    # [0-9] 位置频率特征 (10维)
    # ============================================================

    if n_draws >= 1:
        recent = draws[-30:] if n_draws >= 30 else draws
        # 每个号码在最近30期中的出现频率
        recent_front_nums = Counter()
        recent_back_nums = Counter()
        for d in recent:
            recent_front_nums.update(d[0])
            recent_back_nums.update(d[1])
        freq_front = [recent_front_nums.get(n, 0) / len(recent) for n in front]
        features.extend([
            np.mean(freq_front),        # 0: 前区平均频率
            np.min(freq_front),         # 1: 前区最低频率
            np.max(freq_front),         # 2: 前区最高频率
            np.std(freq_front),         # 3: 前区频率标准差
        ])
        freq_back = [recent_back_nums.get(n, 0) / len(recent) for n in back]
        features.extend([
            np.mean(freq_back),         # 4: 后区平均频率
            np.min(freq_back),          # 5: 后区最低频率
            np.max(freq_back),          # 6: 后区最高频率
        ])
        # 号码是否在最近3期出现
        recent3 = draws[-3:] if n_draws >= 3 else draws
        r3_front = set()
        r3_back = set()
        for d in recent3:
            r3_front.update(d[0])
            r3_back.update(d[1])
        features.extend([
            sum(1 for n in front if n in r3_front) / 5.0,  # 7: 前区近3期重号率
            sum(1 for n in back if n in r3_back) / 2.0,    # 8: 后区近3期重号率
        ])
    else:
        features.extend([0.0] * 9)

    # 上期重号数
    if n_draws >= 2:
        prev_front = set(draws[-1][0])
        features.append(
            sum(1 for n in front if n in prev_front) / 5.0  # 9: 与上期前区重号率
        )
    else:
        features.append(0.0)

    # ============================================================
    # [10-19] 冷热号特征 (10维)
    # ============================================================

    if n_draws >= 1:
        # 最近30期最热的5个号码
        hc_all = Counter()
        for d in draws[-max(30, min(100, n_draws)):]:
            hc_all.update(d[0])
        hot5 = {n for n, _ in hc_all.most_common(5)}
        # 最近50期（或全部）最冷的5个号码
        cold_window = min(50, n_draws)
        cold_all = Counter()
        for d in draws[-cold_window:]:
            cold_all.update(d[0])
        cold5 = set(range(1, 36)) - set(k for k, v in cold_all.items() if v > 0)
        if len(cold5) < 5:
            # 补充遗漏值最高的号码
            sorted_by_freq = sorted(cold_all.items(), key=lambda x: x[1])
            extra_cold = [n for n, _ in sorted_by_freq[:10]]
            cold5 = set(list(cold5)[:3] + extra_cold[:5])
            cold5 = set(list(cold5)[:5])

        features.extend([
            sum(1 for n in front if n in hot5) / 5.0,   # 10: 前区热号含率
            sum(1 for n in front if n in cold5) / 5.0,  # 11: 前区冷号含率
            sum(1 for n in front if n in PRIMES_FRONT) / 5.0,  # 12: 质数占比
        ])
    else:
        features.extend([0.0, 0.0, 0.0])

    # 遗漏值
    if n_draws >= 10:
        omissions = {}
        for n in range(1, 36):
            omit = 0
            for d in reversed(draws):
                if n in d[0]:
                    break
                omit += 1
            omissions[n] = omit
        front_omissions = [omissions.get(n, 0) for n in front]
        features.extend([
            np.mean(front_omissions) / max(n_draws, 1),   # 13: 前区平均遗漏
            np.min(front_omissions) / max(n_draws, 1),    # 14: 前区最小遗漏
            np.max(front_omissions) / max(n_draws, 1),    # 15: 前区最大遗漏
            np.std(front_omissions) / max(n_draws, 1) if np.std(front_omissions) > 0 else 0.0,  # 16
        ])
        back_omissions = {}
        for n in range(1, 13):
            omit = 0
            for d in reversed(draws):
                if n in d[1]:
                    break
                omit += 1
            back_omissions[n] = omit
        back_o = [back_omissions.get(n, 0) for n in back]
        features.extend([
            np.mean(back_o) / max(n_draws, 1),            # 17: 后区平均遗漏
            np.min(back_o) / max(n_draws, 1),             # 18: 后区最小遗漏
        ])
    else:
        features.extend([0.0] * 6)

    # 后区最大遗漏
    if n_draws >= 10:
        features.append(np.max([back_omissions.get(n, 0) for n in back]) / max(n_draws, 1))  # 19
    else:
        features.append(0.0)

    # ============================================================
    # [20-29] 区域分布特征 (10维)
    # ============================================================

    z1 = sum(1 for n in front if n in ZONE1) / 5.0    # 20: 一区占比
    z2 = sum(1 for n in front if n in ZONE2) / 5.0    # 21: 二区占比
    z3 = sum(1 for n in front if n in ZONE3) / 5.0    # 22: 三区占比
    features.extend([z1, z2, z3])

    # 区间重心 (1.0=全一区, 3.0=全三区)
    zone_gravity = (z1 * 1.0 + z2 * 2.0 + z3 * 3.0) / max(z1 + z2 + z3, 0.01)
    features.append(zone_gravity)                      # 23: 区间重心

    # 散度特征
    if len(front) >= 2:
        span = max(front) - min(front)
        gap_min = min(front[i+1] - front[i] for i in range(len(front)-1))
        gap_max = max(front[i+1] - front[i] for i in range(len(front)-1))
    else:
        span, gap_min, gap_max = 0, 0, 0
    features.extend([
        span / 34.0,                                  # 24: 跨度
        gap_min / 10.0,                                # 25: 最小间距
        gap_max / 20.0,                                # 26: 最大间距
        compute_scatter(front),                        # 27: 散度
    ])

    # 区间分布均匀度
    zone_entropy = 0.0
    for p in [z1, z2, z3]:
        if p > 0:
            zone_entropy -= p * np.log(p)
    features.append(zone_entropy / np.log(3))          # 28: 区间熵（归一化）

    # 后区区间
    back_z1 = sum(1 for n in back if n <= 6) / 2.0
    back_z2 = sum(1 for n in back if n >= 7) / 2.0
    features.append(back_z1)                           # 29: 后区小号占比

    # ============================================================
    # [30-39] 数学结构特征 (10维)
    # ============================================================

    fsum = sum(front)
    bsum = sum(back)

    # 和值特征
    if n_draws >= 10:
        recent_sums = [sum(d[0]) for d in draws[-10:]]
        avg_sum = np.mean(recent_sums)
        std_sum = np.std(recent_sums)
    else:
        avg_sum, std_sum = 100, 20
    features.extend([
        fsum / 150.0,                                  # 30: 前区和值标准化
        (fsum - avg_sum) / max(std_sum, 1),            # 31: 和值Z-score
    ])

    # 奇偶比
    front_odd = sum(1 for n in front if n % 2 == 1) / 5.0
    back_odd = sum(1 for n in back if n % 2 == 1) / 2.0
    features.extend([
        front_odd,                                     # 32: 前区奇偶比(奇数占比)
        back_odd,                                      # 33: 后区奇偶比
        abs(front_odd - 0.6),                          # 34: 前区奇偶偏离(理想3:2)
    ])

    # AC值
    ac = ac_value(front)
    features.append(min(ac, 15) / 15.0)                # 35: AC值标准化

    # 尾号聚合
    tail_d = compute_tail_density(front)
    features.append(tail_d)                            # 36: 尾号密度

    # 连号特征
    consec = sum(1 for i in range(len(front)-1) if front[i+1] - front[i] == 1)
    features.append(min(consec, 3) / 3.0)              # 37: 连号数

    # 012路
    road0 = sum(1 for n in front if n % 3 == 0) / 5.0
    road1 = sum(1 for n in front if n % 3 == 1) / 5.0
    road2 = sum(1 for n in front if n % 3 == 2) / 5.0
    features.append(road0)                             # 38: 0路占比
    features.append(road1)                             # 39: 1路占比
    # (2路 = 1 - road0 - road1，可通过3维信息重建)

    # ============================================================
    # [40-49] 跨期模式特征 (10维)
    # ============================================================

    if n_draws >= 2:
        prev = draws[-1]
        prev2 = draws[-2] if n_draws >= 3 else draws[-1]

        # 与上期的模式匹配
        fsum_prev = sum(prev[0])
        fsum_diff = fsum - fsum_prev
        features.append(fsum_diff / 50.0)              # 40: 和值差标准化

        # 隔期重号
        skip_repeat = sum(1 for n in front if n in prev2[0]) / 5.0
        features.append(skip_repeat)                    # 41: 隔期重号率

        # 位置对匹配（前区）
        pos_matches = 0
        for i in range(5):
            if front[i] == prev[0][i] if i < len(prev[0]) else False:
                pos_matches += 1
        features.append(pos_matches / 5.0)              # 42: 位置匹配率

        # 跨度延续性
        span_prev = max(prev[0]) - min(prev[0])
        span_diff = abs(span - span_prev)
        features.append(span_diff / 30.0)               # 43: 跨度差

        # 后区延续
        back_repeat = sum(1 for n in back if n in prev[1]) / 2.0
        features.append(back_repeat)                    # 44: 后区重号率

        # 跨前区后区的关联（上期前区最大→本期后区）
        if n_draws >= 3:
            p2_front_max = max(prev2[0])
            features.append(
                sum(1 for n in back if abs(n - p2_front_max % 12) <= 2) / 2.0
            )                                           # 45: 跨期后区关联
        else:
            features.append(0.0)
    else:
        features.extend([0.0] * 6)

    # 奇偶比模式延续
    odd_prev = sum(1 for n in draws[-1][0] if n % 2 == 1) / 5.0 if n_draws >= 1 else 0.5
    features.append(abs(front_odd - odd_prev))         # 46: 奇偶变化

    # AC值变化
    if n_draws >= 2:
        ac_prev = ac_value(draws[-1][0])
        features.append(abs(ac - ac_prev) / 10.0)       # 47: AC变化
    else:
        features.append(0.0)

    # 三区分布是否与上期相似
    if n_draws >= 2:
        prev_z1 = len(set(draws[-1][0]) & ZONE1) / 5.0
        prev_z2 = len(set(draws[-1][0]) & ZONE2) / 5.0
        prev_z3 = len(set(draws[-1][0]) & ZONE3) / 5.0
        zone_sim = 1.0 - (abs(z1 - prev_z1) + abs(z2 - prev_z2) + abs(z3 - prev_z3)) / 2.0
        features.append(zone_sim)                       # 48: 区间分布相似度
    else:
        features.append(0.0)

    # 尾号变化
    if n_draws >= 2:
        prev_tails = set(n % 10 for n in draws[-1][0])
        cur_tails = set(n % 10 for n in front)
        tail_overlap = len(cur_tails & prev_tails) / max(len(cur_tails | prev_tails), 1)
        features.append(tail_overlap)                   # 49: 尾号延续
    else:
        features.append(0.0)

    # ============================================================
    # [50-59] 元特征 (10维) — 策略/池归属 + 现有评分
    # ============================================================

    # 策略名称one-hot简化版
    strategy = candidate.get('strategy_name', '')
    strat_map = {
        'SFE': 0, 'MultiPoolSampler': 1, '模式池-PatternPool': 2,
        'PoolSampler': 3, 'paired_repeat': 4, 'zero_repeat': 5,
    }
    strat_code = strat_map.get(strategy, 0)
    features.append(strat_code / 5.0)                   # 50: 策略编码

    features.append(candidate.get('base_score', 0.5))    # 51: 基准评分
    features.append(candidate.get('gt_score', 0.5))      # 52: 博弈论评分
    features.append(candidate.get('genetic_score', 0.5))  # 53: 遗传评分

    # 方案B的重号增强标记
    marked = 1.0 if candidate.get('skip_repeat_boosted', False) else 0.0
    features.append(marked)                              # 54: 隔期重号标记

    # ============================================================
    # 【方向B】交叉特征 (12维, index 55-66)
    # ============================================================
    cross_feat = _compute_cross_features(front, back, draws)
    features.extend(cross_feat)

    # 检查特征数量
    expected = 66
    if len(features) != expected:
        print(f"[FEATURE] ⚠️ 特征维度异常: {len(features)} (期望{expected})")
        while len(features) < expected:
            features.append(0.0)

    return features[:expected]


# ============================================================
# 【缺口2】特征重要性跟踪与衰减
# ============================================================

class FeatureDecayer:
    """
    动态特征衰减器 — 追踪每个特征在近期的有效性，低效特征自动休眠。

    每期开奖后更新:
    1. 对每个特征计算"Top10 vs Bottom10区分度"
    2. 区分度低于阈值的标记为休眠
    3. 休眠特征从训练输入中移除（但保留在推理中置零）
    """

    def __init__(self, n_features: int = 67, decay_window: int = 20,
                 discriminance_threshold: float = 0.15):
        self.n_features = n_features
        self.decay_window = decay_window
        self.threshold = discriminance_threshold
        self.history = []  # [{feature_idx: discriminance}, ...]
        self.sleeping = set()  # 休眠特征索引
        self.scores = {i: [] for i in range(n_features)}

    def update(self, candidates_features: List[List[float]],
               top_k: int = 10):
        """
        根据当前候选特征更新区分度评分。

        Args:
            candidates_features: 所有候选的特征向量列表
            top_k: 取前k个为"高分候选"
        """
        if len(candidates_features) < top_k * 2:
            return

        import numpy as np
        arr = np.array(candidates_features)
        top = arr[:top_k]
        bottom = arr[-top_k:]

        discriminances = {}
        for i in range(self.n_features):
            t_mean = float(np.mean(top[:, i]))
            b_mean = float(np.mean(bottom[:, i]))
            std = float(np.std(arr[:, i])) + 1e-8
            d = abs(t_mean - b_mean) / std
            discriminances[i] = d

            if len(self.history) < self.decay_window:
                self.scores[i].append(d)
            else:
                # 滑动窗口更新
                self.scores[i] = self.scores[i][-(self.decay_window-1):] + [d]

        # 判断休眠状态
        self.sleeping = set()
        for i in range(self.n_features):
            recent = self.scores[i][-min(len(self.scores[i]), 5):]
            avg_d = float(np.mean(recent)) if recent else 0.0
            if avg_d < self.threshold:
                self.sleeping.add(i)

        self.history.append(discriminances)

    def get_active_mask(self) -> List[bool]:
        """返回特征激活掩码，True=激活"""
        return [i not in self.sleeping for i in range(self.n_features)]

    def get_sleeping_features(self) -> List[int]:
        """返回当前休眠的特征索引"""
        return sorted(self.sleeping)

    def summary(self) -> str:
        active = self.n_features - len(self.sleeping)
        if self.sleeping:
            return f"特征衰减: {active}/{self.n_features}激活, 休眠={sorted(self.sleeping)[:5]}..."
        return f"特征衰减: {active}/{self.n_features}激活 (全部激活)"


# 全局特征衰减器实例
_feature_decayer = FeatureDecayer()


def get_feature_decayer() -> FeatureDecayer:
    return _feature_decayer


def apply_feature_decay(features: List[float]) -> List[float]:
    """
    将已休眠的特征置零（不参与模型训练/推理）。
    """
    mask = _feature_decayer.get_active_mask()
    return [f if mask[i] else 0.0 for i, f in enumerate(features)]




def compute_match_label(
    candidate_front: List[int],
    candidate_back: List[int],
    actual_front: List[int],
    actual_back: List[int],
) -> float:
    """
    计算候选人命中标签（0.0~5.0），作为ranking label。

    分值 = 前区命中数 + 后区命中数 × 0.5
    """
    front_hits = len(set(candidate_front) & set(actual_front))
    back_hits = len(set(candidate_back) & set(actual_back))
    return float(front_hits + back_hits * 0.5)


def compute_weighted_label(
    candidate_front: List[int],
    candidate_back: List[int],
    actual_front: List[int],
    actual_back: List[int],
) -> float:
    """
    【限制3】成本感知标签 — 按奖项等级加权

    加权规则（基于体彩大乐透奖项）:
    - 前区命中5个 + 后区命中2个 = 一等奖  → 权重 5.0
    - 前区命中5个 + 后区命中1个 = 二等奖  → 权重 4.0
    - 前区命中5个 + 后区命中0个 = 三等奖  → 权重 3.0
    - 前区命中4个 = 四/五等奖       → 权重 2.0
    - 前区命中3个 = 六等奖         → 权重 1.0
    - 前区命中0-2个 = 无奖        → 权重 0.3

    返回值: 0.0 ~ 5.0
    """
    front_hits = len(set(candidate_front) & set(actual_front))
    back_hits = len(set(candidate_back) & set(actual_back))

    if front_hits == 5 and back_hits == 2:
        return 5.0  # 一等奖
    elif front_hits == 5 and back_hits == 1:
        return 4.0  # 二等奖
    elif front_hits == 5:
        return 3.0  # 三等奖
    elif front_hits == 4:
        return 2.0  # 四/五等奖
    elif front_hits == 3:
        return 1.0  # 六等奖
    else:
        return float(front_hits + back_hits * 0.5) * 0.3  # 无奖，降权


FEATURE_NAMES = [
    # 0-9: 位置频率
    'freq_mean', 'freq_min', 'freq_max', 'freq_std',
    'bfreq_mean', 'bfreq_min', 'bfreq_max',
    'recent3_ratio_f', 'recent3_ratio_b',
    'prev_repeat_ratio',
    # 10-19: 冷热号
    'hot_ratio', 'cold_ratio', 'prime_ratio',
    'omit_mean', 'omit_min', 'omit_max', 'omit_std',
    'bomit_mean', 'bomit_min', 'bomit_max',
    # 20-29: 区域分布
    'z1_ratio', 'z2_ratio', 'z3_ratio',
    'zone_gravity', 'span', 'gap_min', 'gap_max', 'scatter',
    'zone_entropy', 'back_low_ratio',
    # 30-39: 数学结构
    'sum_norm', 'sum_zscore',
    'parity_f', 'parity_b', 'parity_deviation',
    'ac_norm', 'tail_density',
    'consecutive', 'road0', 'road1',
    # 40-49: 跨期模式
    'sum_diff', 'skip_repeat', 'pos_match', 'span_diff',
    'back_repeat', 'cross_relate', 'parity_change',
    'ac_change', 'zone_sim', 'tail_continuity',
    # 50-54: 元特征
    'strategy_code', 'base_score', 'gt_score', 'genetic_score',
    'skip_repeat_marked',
    # 55-66: 【B】交叉特征
    'freq_x_omit', 'omit_prod',
    'sum_x_span', 'parity_x_road',
    'z1_x_z3', 'repeat_x_cold',
    'ac_x_span', 'tail_x_scatter',
    'consec_x_sum_skew', 'back_odd_x_front_odd',
    'max_x_min_cross',
]

assert len(FEATURE_NAMES) == 66, f"FEATURE_NAMES长度={len(FEATURE_NAMES)}, 期望66"
