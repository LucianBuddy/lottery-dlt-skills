#!/usr/bin/env python3
"""
DLT 集成投票和候选池生成模块

从 DLTFusionComplete 中提取的方法：
- _ensemble_vote() → vote()
- _sample_skip_repeat_candidates() → sample_skip_repeat()
- _generate_ts_candidates() → generate_ts_candidates()
- _sample_pool_candidates() → sample_pool()
- _sample_pattern_pool_candidates() → sample_pattern_pool()

所有 self.xxx 引用改为 self.master.xxx
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict


class DLTEnsembleVoter:
    """DLT 多模型集成投票和候选池生成 """

    def __init__(self, master):
        self.master = master  # DLTFusionComplete 实例引用

    def vote(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        【缺口1】多模型集成投票 — 4模型独立评分后加权融合

        参与模型:
        A. GBR ranking model (权重=最近20期MRR)
        B. 决策树 (权重=最近20期MRR)
        C. 时序LR (权重=最近20期MRR)
        D. 频率基线 (权重固定0.5)

        每个模型对候选输出35维概率向量，
        候选得分 = sum(模型权重 × 模型输出的该候选5个号码平均概率)。

        Returns:
            更新final_score后的候选列表
        """
        master = self.master
        mem = getattr(master, '_mem_mb', 800)
        if mem < 300:
            return candidates  # 低内存跳过

        try:
            n_draws = len(master.draws)
            if n_draws < 20:
                return candidates

            from modules.ranking_feature_extractor import extract_features, ac_value

            prev_front = master.draws[-1][0]
            prev_sum = sum(prev_front)
            prev_span = max(prev_front) - min(prev_front)
            prev_odd = sum(1 for n in prev_front if n % 2 == 1) / 5.0
            prev_z1 = len([n for n in prev_front if n <= 12]) / 5.0
            prev_z2 = len([n for n in prev_front if 13 <= n <= 24]) / 5.0
            prev_z3 = len([n for n in prev_front if n >= 25]) / 5.0
            prev_ac = ac_value(prev_front) / 15.0

            # ============================================================
            # 模型A: GBR ranking model → 35维概率
            # ============================================================
            prob_a = np.ones(35) * 0.5
            weight_a = 0.0
            if master.ranking_model is not None and master.ranking_model.is_trained:
                try:
                    for num in range(1, 36):
                        dummy = {'front': [num] + list(range(1, 6) if num > 5 else range(6, 11)),
                                 'back': [1, 12], 'base_score': 0.5, 'strategy_name': 'SFE'}
                        feat = extract_features(dummy, master.draws, master._periods)
                        score = master.ranking_model.predict_single(feat)
                        prob_a[num-1] = float(score)
                    prob_a = prob_a / (np.sum(prob_a) + 1e-8) * 5.0
                    weight_a = 0.35
                except Exception:
                    weight_a = 0.0

            # ============================================================
            # 模型B: 决策树 → 35维概率
            # ============================================================
            prob_b = np.ones(35) * 0.5
            weight_b = 0.0
            if hasattr(master, '_dt_model') and master._dt_model is not None:
                try:
                    x_feat = np.array([[prev_sum/150.0, prev_span/34.0, prev_odd,
                                        prev_ac, prev_z1, prev_z2, prev_z3]])
                    prob_b = master._dt_model.predict(x_feat)[0]
                    weight_b = 0.25
                except Exception:
                    weight_b = 0.0

            # ============================================================
            # 模型C: 时序LR → 35维概率
            # ============================================================
            prob_c = np.ones(35) * 0.5
            weight_c = 0.0
            try:
                from sklearn.linear_model import LogisticRegression
                window = min(50, n_draws - 1)
                X_c, y_c = [], []
                for i in range(window - 5, n_draws - 1):
                    feats = []
                    for j in range(5):
                        d = master.draws[i - 4 + j]
                        fs = sum(d[0])
                        feats.extend([fs/150.0, (max(d[0])-min(d[0]))/34.0,
                                      sum(1 for n in d[0] if n%2==1)/5.0])
                    freq = Counter()
                    for d in master.draws[max(0,i-29):i+1]:
                        freq.update(d[0])
                    feats.extend([freq.get(n,0)/30.0 for n in range(1,36)])
                    recent_seq = np.zeros(35)
                    for d in master.draws[max(0,i-19):i+1]:
                        for n in d[0]:
                            recent_seq[n-1] += 1
                    recent_seq = recent_seq / max(20, 1)
                    feats.extend(recent_seq.tolist())
                    X_c.append(feats)
                    next_d = master.draws[i+1]
                    y_c.append([1.0 if n in next_d[0] else 0.0 for n in range(1,36)])

                if len(X_c) >= 20:
                    X_arr = np.array(X_c)
                    y_arr = np.array(y_c)
                    for num in range(35):
                        clf = LogisticRegression(max_iter=200, C=0.5, solver='lbfgs')
                        clf.fit(X_arr, y_arr[:, num])
                        prob_c[num] = clf.predict_proba(X_arr[-1:].reshape(1, -1))[0][1]
                    weight_c = 0.25
            except Exception:
                weight_c = 0.0

            # ============================================================
            # 模型D: 频率基线 (固定权重0.15)
            # ============================================================
            freq_d = Counter()
            for d in master.draws[-30:]:
                freq_d.update(d[0])
            prob_d = np.array([freq_d.get(n, 0) / 30.0 for n in range(1, 36)])
            weight_d = 0.15

            # ============================================================
            # 加权平均 → 候选评分
            # ============================================================
            total_weight = weight_a + weight_b + weight_c + weight_d
            if total_weight < 0.3:
                return candidates

            prob_final = (prob_a * weight_a + prob_b * weight_b +
                          prob_c * weight_c + prob_d * weight_d) / total_weight

            boost_count = 0
            for c in candidates:
                front = c.get('front', [])
                ensemble = float(np.mean([prob_final[n-1] for n in front]))
                c['ensemble_score'] = ensemble
                orig = c.get('final_score', 0.5)
                c['final_score'] = orig * 0.75 + ensemble * 0.25
                boost_count += 1

            weights_str = f"A={weight_a:.2f} B={weight_b:.2f} C={weight_c:.2f} D={weight_d:.2f}"
            print(f"[DLT-Fusion] 🎯 【1】集成投票({weights_str}): "
                  f"{boost_count}候选更新, 最高ensemble={float(np.max(prob_final)):.3f}")

        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 【1】集成投票跳过: {e}")

        return candidates

    def sample_skip_repeat(self, n_candidates: int = 4) -> List[Dict[str, Any]]:
        """
        方案B：双期重号参考候选池
        基于draws[-1]和draws[-2]的重号模式生成候选，覆盖隔期回归号码。

        Args:
            n_candidates: 生成候选数量，默认4

        Returns:
            List[Dict]: 候选列表
        """
        master = self.master
        draws = master.draws

        if len(draws) < 3:
            return []

        candidates = []
        prev_front = draws[-1][0]
        prev_back = draws[-1][1]
        skip_front = draws[-2][0]
        skip_back = draws[-2][1]

        core_front = list(set(prev_front + skip_front))
        core_back = list(set(prev_back + skip_back))

        if len(core_front) < 5:
            extra = [n for n in range(1, 36) if n not in core_front]
            random.shuffle(extra)
            core_front.extend(extra[:10 - len(core_front)])
        if len(core_back) < 2:
            extra = [n for n in range(1, 13) if n not in core_back]
            random.shuffle(extra)
            core_back.extend(extra[:4 - len(core_back)])

        strategies = []

        # 类型1：隔期回归型
        for _ in range(n_candidates // 2 + 1):
            base = list(skip_front)
            replace_i = random.randrange(len(base))
            base[replace_i] = random.choice(prev_front)
            back = sorted(random.sample(core_back, min(2, len(core_back))))
            strategies.append((sorted(base), back, 'skip_repeat_ago', 0.65))

        # 类型2：立即重号型
        for _ in range(n_candidates // 2 + 1):
            base = list(prev_front)
            replace_i = random.randrange(len(base))
            base[replace_i] = random.choice(skip_front)
            back = sorted(random.sample(core_back, min(2, len(core_back))))
            strategies.append((sorted(base), back, 'skip_repeat_prev', 0.60))

        # 类型3：混合型
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

    def generate_ts_candidates(self) -> List[Dict[str, Any]]:
        """
        【方向1】号码时序预测 — 多步线性回归预测下一期号码概率

        将35个前区号码看作35维时间序列，
        从概率最高的16个号码中枚举C(16,5)=4368组合作为候选。

        在内存不足(<300MB)时自动跳过。
        """
        master = self.master
        mem = getattr(master, '_mem_mb', 800)
        if mem < 300:
            return []

        try:
            n_draws = len(master.draws)
            if n_draws < 30:
                return []

            from sklearn.linear_model import LogisticRegression
            import itertools
            import warnings
            warnings.filterwarnings('ignore')

            window = min(50, n_draws - 1)

            X, y = [], []
            for i in range(window - 5, n_draws - 1):
                feats = []
                for j in range(5):
                    d = master.draws[i - 4 + j]
                    fs = sum(d[0])
                    feats.extend([fs / 150.0,
                                  (max(d[0]) - min(d[0])) / 34.0,
                                  sum(1 for n in d[0] if n % 2 == 1) / 5.0])

                freq = Counter()
                for d in master.draws[max(0, i - 29):i+1]:
                    freq.update(d[0])
                feats.extend([freq.get(n, 0) / 30.0 for n in range(1, 36)])

                recent_seq = np.zeros(35)
                for d in master.draws[max(0, i - 19):i+1]:
                    for n in d[0]:
                        recent_seq[n-1] += 1
                recent_seq = recent_seq / max(20, 1)
                feats.extend(recent_seq.tolist())

                X.append(feats)
                next_d = master.draws[i + 1]
                y_vec = [1.0 if n in next_d[0] else 0.0 for n in range(1, 36)]
                y.append(y_vec)

            if len(X) < 20:
                return []

            X_arr = np.array(X)
            y_arr = np.array(y)

            probs = np.zeros(35)
            for num in range(35):
                clf = LogisticRegression(max_iter=200, C=0.5, solver='lbfgs')
                clf.fit(X_arr, y_arr[:, num])
                probs[num] = clf.predict_proba(X_arr[-1:].reshape(1, -1))[0][1]

            top16 = sorted(range(1, 36), key=lambda n: -probs[n-1])[:16]

            candidates = []
            back_pool = master.get_back_recommendations()
            back_pool = back_pool if back_pool else [[1, 12]]

            for combo in itertools.combinations(top16, 5):
                front = sorted(combo)
                back = back_pool[len(candidates) % len(back_pool)]
                candidates.append({
                    'front': front,
                    'back': back,
                    'base_score': float(np.mean([probs[n-1] for n in front])),
                    'strategy_name': 'TimeSeries-LR',
                    'source': 'ts_predict',
                })

            candidates.sort(key=lambda x: -x['base_score'])
            result = candidates[:20]

            print(f"[DLT-Fusion] 📈 【1】时序预测: top16={top16[:8]}... "
                  f"({len(candidates)}候选→选20注)")

            return result

        except Exception as e:
            print(f"[DLT-Fusion] ⚠️ 【1】时序预测跳过: {e}")
            return []

    def sample_pool(self, n_per_pool: int = 2) -> List[Dict[str, Any]]:
        """多池采样候选 — [修复] 后区使用back_fusion推荐而非random"""
        master = self.master
        candidates = []
        try:
            front_combos = master.pool_sampler.stratified_sample(
                n_combinations=n_per_pool * 3, zone='front'
            )
            back_recs_local = master.get_back_recommendations()
            if not back_recs_local:
                back_recs_local = [[1, 6], [5, 11], [3, 8], [4, 10], [2, 12]]

            for i, fc in enumerate(front_combos[:n_per_pool * 2]):
                bc = back_recs_local[i % len(back_recs_local)]
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

    def sample_pattern_pool(self) -> List[Dict[str, Any]]:
        """模式池采样：基于跨期模式识别生成候选"""
        master = self.master
        candidates = []
        try:
            front_pool, back_pool = master.pattern_recognizer.generate_pattern_pool(
                n_front=12, n_back=4
            )
            for i in range(3):
                selected_front = sorted(random.sample(front_pool, 5))
                selected_back = sorted(random.sample(back_pool, 2))
                candidates.append({
                    'front': selected_front,
                    'back': selected_back,
                    'source': 'pattern_pool',
                    'total_score': 0.52,
                    'strategy_name': '模式池-PatternPool',
                })
        except Exception as e:
            print(f"[DLT-Fusion] 模式池采样失败: {e}")

        return candidates
