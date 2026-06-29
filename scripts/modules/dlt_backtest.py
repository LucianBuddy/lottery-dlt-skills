#!/usr/bin/env python3
"""
DLT 回测和分析模块

从 DLTFusionComplete 中提取的方法：
- backtest() → run()
- post_draw_analysis() → post_draw_analysis()

所有 self.xxx 引用改为 self.master.xxx
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter


class DLTBacktest:
    """DLT 回测和分析 — 测量命中率、覆盖率、三基线对比、反事实分析"""

    def __init__(self, master):
        self.master = master  # DLTFusionComplete 实例引用

    def run(self, n_recent: int = 100) -> Dict[str, Any]:
        """
        重建回测：测量池级别命中率 vs 随机基准

        核心指标：
        - 每个池的平均命中个数（应该 > 随机基准 才算有效）
        - 每种复式类型的覆盖率（开奖号码在复式中的比例）
        - 对比随机基准：模型提升百分比
        """
        import random as _rnd
        _rnd.seed(42)

        draws = self.master.draws

        if len(draws) < n_recent + 20:
            return {'error': f'数据不足 (需要{n_recent+20}期, 现有{len(draws)}期)'}

        test_draws = draws[-n_recent:]
        train_base = draws[:len(draws) - n_recent]

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

        # ================================================================
        # 【方案5】三基线对比：随机/热号/模式 + model beats all
        # ================================================================
        try:
            from lottery_metrics import LotteryMetrics
            met = LotteryMetrics()
            mrr_scores = []
            ndcg_scores = []
            ece_bins = {f'b{i}': {'pred': 0, 'actual': 0} for i in range(1, 11)}
            hot_baseline_hits = []
            pattern_baseline_hits = []
            model_beats_all = []  # 每期模型>全部基线的标记
            model_total = []      # 模型本身的命中数/期

            for i, actual in enumerate(test_draws):
                hist = train_base + test_draws[:i] if i > 0 else train_base
                sampler = MultiPoolSampler(hist)
                actual_front = set(actual[0])

                # --- hot_pool打分（模型评分基准） ---
                pool_best = sampler.generate_hot_pool(20, 'front')
                for rank, n in enumerate(pool_best, 1):
                    if n in actual_front:
                        mrr_scores.append(1.0 / rank)
                        break
                else:
                    mrr_scores.append(0.0)

                ndcg = met.ndcg_at_k(pool_best, list(actual_front), k=5)
                ndcg_scores.append(ndcg)

                # ECE
                high_conf = set(pool_best[:5])
                ece_bins['b10']['pred'] += 5
                ece_bins['b10']['actual'] += len(high_conf & actual_front)
                ece_bins['b2']['pred'] += 15
                low_hits = len(set(pool_best[5:]) & actual_front)
                ece_bins['b2']['actual'] += low_hits

                # --- 基线1: 热号基线（近期最热5号） ---
                recent_nums = Counter()
                for d in hist[-30:]:
                    recent_nums.update(d[0])
                hot_baseline = [n for n, _ in recent_nums.most_common(5)]
                hot_hits = len(set(hot_baseline) & actual_front)
                hot_baseline_hits.append(hot_hits)

                # --- 基线2: 模式基线（上期+上上期重号） ---
                pattern_pool = []
                if len(hist) >= 1:
                    pattern_pool.extend(hist[-1][0])
                if len(hist) >= 2:
                    pattern_pool.extend(hist[-2][0])
                # 去重后取前5个
                pattern_baseline = list(dict.fromkeys(pattern_pool))[:5]
                # 如果不足5个，补随机号码
                while len(pattern_baseline) < 5:
                    import random
                    r = random.randint(1, 35)
                    if r not in pattern_baseline:
                        pattern_baseline.append(r)
                pattern_hits = len(set(pattern_baseline) & actual_front)
                pattern_baseline_hits.append(pattern_hits)

                # --- 基线3: 随机基线（5个随机号码） ---
                rand_hits = sum(
                    1 for d in [[random.randint(1,35) for _ in range(5)] for _ in range(30)]
                    if len(set(d) & actual_front) > 0
                ) / 30.0

                # --- 模型实际命中（从pool_best取前5） ---
                model_hits = len(set(pool_best[:5]) & actual_front)
                model_total.append(model_hits)

                # --- 判断: 模型是否跑赢全部基线 ---
                beats = (model_hits > hot_hits and
                         model_hits > pattern_hits and
                         model_hits > rand_hits)
                model_beats_all.append(1 if beats else 0)

            # MRR
            avg_mrr = round(float(np.mean(mrr_scores)), 4)

            # NDCG
            avg_ndcg = round(float(np.mean(ndcg_scores)), 4)

            # ECE
            ece_vals = []
            for bname, bdata in ece_bins.items():
                if bdata['pred'] > 0:
                    acc = bdata['actual'] / bdata['pred']
                    conf = 0.95 if 'b10' in bname else 0.25
                    ece_vals.append(abs(acc - conf))
            avg_ece = round(float(np.mean(ece_vals)), 4) if ece_vals else 0.0

            # 热号基线
            avg_hot_base = round(float(np.mean(hot_baseline_hits)), 3)
            hot_base_imp = round(
                (avg_hot_base / random_front_expect - 1) * 100, 1
            ) if random_front_expect > 0 else 0

            # 模式基线
            avg_pattern_base = round(float(np.mean(pattern_baseline_hits)), 3)
            pattern_base_imp = round(
                (avg_pattern_base / random_front_expect - 1) * 100, 1
            ) if random_front_expect > 0 else 0

            # 模型本身
            avg_model = round(float(np.mean(model_total)), 3)

            # model beats all rate
            beats_rate = round(sum(model_beats_all) / max(len(model_beats_all), 1) * 100, 1)

            # 汇总
            result['enhanced_metrics'] = {
                'mrr': avg_mrr,
                'ndcg_at_5': avg_ndcg,
                'ece': avg_ece,
                'baselines': {
                    'random': {'label': '随机5号', 'avg_hits_per_draw': round(random_front_expect, 3)},
                    'hot': {'label': '热号(近30期最热5号)', 'avg_hits_per_draw': avg_hot_base,
                            'vs_random_imp_%': hot_base_imp},
                    'pattern': {'label': '模式(上期+上上期重号)', 'avg_hits_per_draw': avg_pattern_base,
                                'vs_random_imp_%': pattern_base_imp},
                },
                'model': {
                    'avg_hits_per_draw': avg_model,
                    'beats_all_baselines_%': beats_rate,
                },
                'description': {
                    'mrr': '平均倒数排名(越高越好, 随机≈0.14)',
                    'ndcg_at_5': '归一化折损累计增益@5(越高越好, 随机≈0.5)',
                    'ece': '期望校准误差(越低越好, 完美=0)',
                    'model_beats_all': '模型跑赢全部基线的期数占比',
                }
            }

            print(f"[DLT-Metrics] 📊 【方案5】MRR={avg_mrr:.4f} NDCG={avg_ndcg:.4f} "
                  f"ECE={avg_ece:.4f} model={avg_model:.3f}/注 "
                  f"热号={avg_hot_base:.3f} 模式={avg_pattern_base:.3f} "
                  f"beats_all={beats_rate}%")
        except Exception as e:
            print(f"[DLT-Metrics] ⚠️ 【方案5】基线对比跳过: {e}")

        return result

    def post_draw_analysis(self, actual_front: List[int], actual_back: List[int]) -> dict:
        """
        【方向3】逆强化反馈 — 开奖后反事实分析

        在每期开奖后调用，分析模型"哪里错了":
        1. 遗憾排名: 实际号码在候选池中的排序位置
        2. 降分回溯: 实际号码在评分pipeline的哪一步被降分最多
        3. 偏差归因: 模型偏向(热号偏执/冷号恐惧/和值区间)

        Args:
            actual_front: 实际前区5个号码
            actual_back: 实际后区2个号码

        Returns:
            dict: 诊断报告
        """
        master = self.master

        if not hasattr(master, '_last_candidates') or not master._last_candidates:
            return {'error': '无候选池快照，需先运行predict()'}

        candidates = master._last_candidates
        actual_set = set(actual_front)

        # 1. 遗憾排名 — 实际号码在候选池中的最佳排名
        actual_in_candidates = [c for c in candidates
                                if len(set(c.get('front', [])) & actual_set) >= 3]
        rank_report = {}
        if actual_in_candidates:
            best_rank = min(
                i for i, c in enumerate(candidates)
                if c in actual_in_candidates
            )
            rank_report['best_rank'] = best_rank + 1  # 1-indexed
            rank_report['total_candidates'] = len(candidates)
            rank_report['quality'] = 'good' if best_rank < 10 else 'fair' if best_rank < 30 else 'poor'
        else:
            rank_report['best_rank'] = None
            rank_report['total_candidates'] = len(candidates)
            rank_report['quality'] = 'missing'

        # 2. 偏差归因 — 分析模型当前的偏向性
        draws = master.draws
        recent_10_actual_sums = []
        if len(draws) >= 10:
            for d in draws[-10:]:
                recent_10_actual_sums.append(sum(d[0]))
        else:
            recent_10_actual_sums = [sum(draws[i][0]) for i in range(len(draws))]

        recent_avg_sum = float(np.mean(recent_10_actual_sums)) if recent_10_actual_sums else 100

        # 候选的和值分布 vs 实际和值
        cand_sums = [sum(c.get('front', [])) for c in candidates]
        cand_avg_sum = float(np.mean(cand_sums)) if cand_sums else 100

        # 热号偏执检测
        from collections import Counter
        top5_hot = Counter()
        for d in draws[-30:]:
            top5_hot.update(d[0])
        hottest = {n for n, _ in top5_hot.most_common(5)}
        hot_in_cands = sum(
            1 for c in candidates
            if len(set(c.get('front', [])) & hottest) >= 3
        ) / max(len(candidates), 1)

        # 冷号恐惧检测(遗漏>20期的号码在候选中的比例)
        omissions = {}
        for n in range(1, 36):
            omit = 0
            for d in reversed(draws):
                if n in d[0]:
                    break
                omit += 1
            omissions[n] = omit
        cold_nums = {n for n, o in omissions.items() if o > 20}
        cold_in_cands = sum(
            1 for c in candidates
            if len(set(c.get('front', [])) & cold_nums) >= 2
        ) / max(len(candidates), 1)

        bias_report = {
            'sum_bias': round(cand_avg_sum - recent_avg_sum, 1),
            'hot_bias_pct': round(hot_in_cands * 100, 1),
            'cold_fear_pct': round(cold_in_cands * 100, 1),
            'diagnosis': []
        }
        if abs(bias_report['sum_bias']) > 10:
            bias_report['diagnosis'].append(
                f"和值偏离{cand_avg_sum:.0f} vs 实际近期均值{recent_avg_sum:.0f}"
            )
        if bias_report['hot_bias_pct'] > 60:
            bias_report['diagnosis'].append(f"热号偏执({bias_report['hot_bias_pct']}%候选含≥3热号)")
        if bias_report['cold_fear_pct'] < 5:
            bias_report['diagnosis'].append(f"冷号恐惧(仅{bias_report['cold_fear_pct']}%候选含≥2冷号)")

        # 3. 实际号码的命中详情
        # 用前5个候选与实际的交集
        top5 = candidates[:5]
        top5_hit_detail = []
        for i, c in enumerate(top5):
            front = c.get('front', [])
            hits = len(set(front) & actual_set)
            top5_hit_detail.append({
                'rank': i + 1,
                'front': front,
                'hits': hits,
                'score': round(c.get('final_score', 0), 4),
            })

        report = {
            'regret_rank': rank_report,
            'bias': bias_report,
            'top5_hits': top5_hit_detail,
        }

        # 打印摘要
        quality = rank_report.get('quality', 'unknown')
        bias_diag = '; '.join(bias_report.get('diagnosis', [])) if bias_report.get('diagnosis') else '正常'
        print(f"[DLT-Fusion] 🎯 【3】逆强化反馈: 排名={rank_report.get('best_rank','N/A')}/{rank_report['total_candidates']} "
              f"质量={quality} | {bias_diag}")

        return report
