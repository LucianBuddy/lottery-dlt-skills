#!/usr/bin/env python3
"""
DLT大乐透优化版预测器 V1.0
增强输出：每注概率 + 每种池策略胆拖方案 + 详细评分
"""

import sys, os, json, random, warnings
import numpy as np
from math import factorial
from typing import List, Tuple, Dict, Any, Optional
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dlt_fusion_complete import DLTFusionComplete

def _comb(n: int, k: int) -> int:
    if k < 0 or k > n: return 0
    k = min(k, n - k)
    r = 1
    for i in range(1, k + 1):
        r = r * (n - k + i) // i
    return r

class DLTOptimizedPredictor:
    """优化版预测器，增强概率输出和策略分析"""

    POOL_STRATEGIES = {
        'hot':      {'name': '🔥 热号池',   'desc': '近期高频号码'},
        'cold':     {'name': '❄️ 冷号池',   'desc': '长期遗漏号码'},
        'balance':  {'name': '⚖️ 均衡池',   'desc': '频率居中号码'},
        'trend':    {'name': '📈 趋势池',   'desc': '频率上升号码'},
        'prime':    {'name': '🧬 质数池',   'desc': '质数特征号码'},
    }

    # 胆拖方案配置: {名称: {front_dan:胆码数, front_tuo_min:最小拖码数, ...}}
    DAN_TUO_CONFIGS = [
        {'name': '1胆4拖',  'front_dan': 1, 'front_tuo': 4, 'back_type': '直选',  'back_tuo': 0, 'desc': '1胆4拖 → 4注'},
        {'name': '2胆3拖',  'front_dan': 2, 'front_tuo': 3, 'back_type': '直选',  'back_tuo': 0, 'desc': '2胆3拖 → 1注'},
        {'name': '2胆4拖',  'front_dan': 2, 'front_tuo': 4, 'back_type': '直选',  'back_tuo': 0, 'desc': '2胆4拖 → 4注'},
        {'name': '2胆5拖',  'front_dan': 2, 'front_tuo': 5, 'back_type': '直选',  'back_tuo': 0, 'desc': '2胆5拖 → 10注'},
        {'name': '3胆2拖',  'front_dan': 3, 'front_tuo': 2, 'back_type': '直选',  'back_tuo': 0, 'desc': '3胆2拖 → 1注'},
        {'name': '3胆3拖',  'front_dan': 3, 'front_tuo': 3, 'back_type': '直选',  'back_tuo': 0, 'desc': '3胆3拖 → 3注'},
        {'name': '3胆4拖',  'front_dan': 3, 'front_tuo': 4, 'back_type': '直选',  'back_tuo': 0, 'desc': '3胆4拖 → 6注'},
        {'name': '4胆1拖',  'front_dan': 4, 'front_tuo': 1, 'back_type': '直选',  'back_tuo': 0, 'desc': '4胆1拖 → 1注'},
        {'name': '4胆2拖',  'front_dan': 4, 'front_tuo': 2, 'back_type': '直选',  'back_tuo': 0, 'desc': '4胆2拖 → 2注'},
    ]

    # 后区胆拖选项
    BACK_DAN_TUO_CONFIGS = [
        {'name': '后区直选',   'back_dan': 0, 'back_tuo': 2, 'desc': '选2个后区', 'bets_factor': 1},
        {'name': '后区1胆2拖', 'back_dan': 1, 'back_tuo': 2, 'desc': '1胆2拖 → 2注', 'bets_factor': 2},
        {'name': '后区1胆3拖', 'back_dan': 1, 'back_tuo': 3, 'desc': '1胆3拖 → 3注', 'bets_factor': 3},
        {'name': '后区1胆4拖', 'back_dan': 1, 'back_tuo': 4, 'desc': '1胆4拖 → 4注', 'bets_factor': 4},
    ]

    def __init__(self):
        self.fusion = DLTFusionComplete()
        self.draws = self.fusion.draws
        self._compute_stats()

    def _compute_stats(self):
        """预计算统计特征"""
        self.front_freq = np.zeros(36, dtype=int)
        self.back_freq = np.zeros(13, dtype=int)
        self.front_recent_freq = np.zeros(36, dtype=int)
        self.back_recent_freq = np.zeros(13, dtype=int)

        n = len(self.draws)
        recent_n = min(20, n)
        for i, (front, back) in enumerate(self.draws):
            for f in front:
                self.front_freq[f] += 1
                if i >= n - recent_n:
                    self.front_recent_freq[f] += 1
            for b in back:
                self.back_freq[b] += 1
                if i >= n - recent_n:
                    self.back_recent_freq[b] += 1

        self.total_periods = n
        self.front_freq_pct = self.front_freq / n * 100
        self.back_freq_pct = self.back_freq / n * 100

    def _is_recent_repeat(self, front: List[int], back: List[int]) -> bool:
        if len(self.draws) < 1:
            return False
        return sorted(front) == sorted(self.draws[-1][0])

    def _calc_probability(self, front: List[int], back: List[int]) -> Dict[str, float]:
        front_arr = np.array(front)
        back_arr = np.array(back)
        recent_n = min(20, self.total_periods)

        front_freq_prob = np.mean([self.front_freq[f] / self.total_periods for f in front_arr])
        back_freq_prob = np.mean([self.back_freq[b] / self.total_periods for b in back_arr])
        front_recent_prob = np.mean([self.front_recent_freq[f] / recent_n for f in front_arr])
        back_recent_prob = np.mean([self.back_recent_freq[b] / recent_n for b in back_arr])
        front_missing = [self.total_periods - self.front_freq[f] for f in front_arr]
        back_missing = [self.total_periods - self.back_freq[b] for b in back_arr]
        front_missing_prob = min(np.mean(front_missing) / 100, 1.0)
        back_missing_prob = min(np.mean(back_missing) / 100, 1.0)

        front_sum = np.sum(front_arr)
        sum_score = 1.0 if 80 <= front_sum <= 130 else max(0, 1 - abs(front_sum - 105) / 50)
        odd_count = np.sum(front_arr % 2)
        odd_score = 1.0 - abs(odd_count - 2.5) / 2.5
        span = max(front_arr) - min(front_arr)
        span_score = 1.0 if 15 <= span <= 32 else max(0, 1 - abs(span - 24) / 20)
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
            'front_freq': round(front_freq_prob * 100, 1),
            'front_trend': round(front_recent_prob * 100, 1),
            'front_missing': round(front_missing_prob * 100, 1),
            'math_score': round((sum_score * 0.35 + odd_score * 0.30 + span_score * 0.35) * 100, 1),
        }

    def _generate_dan_tuo(self, pool_name: str) -> List[Dict[str, Any]]:
        """
        为指定策略池生成胆拖投注方案。
        从池中提取高分号码作为胆码，其余作为拖码。
        """
        results = []
        try:
            # 获取该策略池的号码（获取更多号码，以便胆拖有足够选择）
            gen_map = {
                'hot':     (self.fusion.pool_sampler.generate_hot_pool, 12),
                'cold':    (self.fusion.pool_sampler.generate_cold_pool, 12),
                'balance': (self.fusion.pool_sampler.generate_balance_pool, 12),
                'trend':   (self.fusion.pool_sampler.generate_game_theory_pool, 12),
                'prime':   (self.fusion.pool_sampler.generate_game_theory_pool, 12),
            }
            gen_func, pool_size = gen_map.get(pool_name, (self.fusion.pool_sampler.generate_balance_pool, 12))

            pool_front = gen_func(pool_size, 'front')
            pool_back = gen_func(8, 'back')

            # 去重排序
            pool_front = sorted(set(pool_front))
            pool_back = sorted(set(pool_back))

            if len(pool_front) < 5:
                return results

            # 按历史频率排序，选前几个作为胆码候选
            front_scores = [(n, self.front_freq[n]) for n in pool_front]
            front_scores.sort(key=lambda x: -x[1])
            ranked_front = [n for n, _ in front_scores]

            back_scores = [(n, self.back_freq[n]) for n in pool_back]
            back_scores.sort(key=lambda x: -x[1])
            ranked_back = [n for n, _ in back_scores]

            # 生成多种胆拖方案
            # 前区：尝试多种胆码数（1~3个胆码最实用）
            front_dan_configs = [
                {'dan': 1, 'tuo': 4, 'total_bets': _comb(4, 4)},
                {'dan': 2, 'tuo': 3, 'total_bets': _comb(3, 3)},
                {'dan': 2, 'tuo': 4, 'total_bets': _comb(4, 3)},
                {'dan': 3, 'tuo': 2, 'total_bets': _comb(2, 2)},
                {'dan': 3, 'tuo': 3, 'total_bets': _comb(3, 2)},
                {'dan': 1, 'tuo': 5, 'total_bets': _comb(5, 4)},
            ]

            for fdc in front_dan_configs:
                nd = fdc['dan']
                nt = fdc['tuo']
                if len(ranked_front) < nd + nt:
                    continue

                # 胆码 = 排名最高的 nd 个
                dan_nums = ranked_front[:nd]
                # 拖码 = 接下来的 nt 个（跳过胆码）
                tuo_pool = [n for n in ranked_front if n not in dan_nums]
                if len(tuo_pool) < nt:
                    continue
                tuo_nums = tuo_pool[:nt]

                # 检查胆码+拖码是否与上期前区完全相同
                test_front = sorted(dan_nums + tuo_nums)[:5]
                if self._is_recent_repeat(test_front, [1, 2]):
                    continue

                # 后区：直选2个（最推荐）或后区1胆托
                # 方案A: 后区直选
                if len(ranked_back) >= 2:
                    back_nums = ranked_back[:2]
                    if sorted(back_nums) == sorted(self.draws[-1][1]):
                        back_nums = [ranked_back[0], ranked_back[2]] if len(ranked_back) >= 3 else ranked_back[:2]

                    probs = self._calc_probability(dan_nums + tuo_nums[:5-len(dan_nums)], back_nums)
                    total_bets = fdc['total_bets'] * 1  # 后区直选，1种组合
                    name_str = f"{nd}胆{nt}拖+后区直选"

                    results.append({
                        'name': name_str,
                        'front_dan': sorted(dan_nums),
                        'front_tuo': sorted(tuo_nums),
                        'back': sorted(back_nums),
                        'back_desc': '直选',
                        'probability': probs['combined'],
                        'front_prob': probs['front'],
                        'back_prob': probs['back'],
                        'total_bets': total_bets,
                    })

                # 方案B: 后区1胆1拖（当有足够后区号时）
                if len(ranked_back) >= 3:
                    back_dan = ranked_back[:1]
                    back_tuo_pool = [n for n in ranked_back if n not in back_dan]
                    if len(back_tuo_pool) >= 2:
                        back_tuo = back_tuo_pool[:2]
                        test_back = back_dan + back_tuo
                        probs2 = self._calc_probability(dan_nums + tuo_nums[:5-len(dan_nums)], test_back)
                        total_bets2 = fdc['total_bets'] * 2  # 后区1胆2拖 → 2注
                        name_str2 = f"{nd}胆{nt}拖+后区1胆{len(back_tuo)}拖"

                        results.append({
                            'name': name_str2,
                            'front_dan': sorted(dan_nums),
                            'front_tuo': sorted(tuo_nums),
                            'back': sorted(back_dan + back_tuo),
                            'back_desc': f"胆{back_dan}拖{back_tuo}",
                            'probability': probs2['combined'],
                            'front_prob': probs2['front'],
                            'back_prob': probs2['back'],
                            'total_bets': total_bets2,
                        })

        except Exception as e:
            print(f"[胆拖] {pool_name} 生成失败: {e}")

        results.sort(key=lambda x: x['probability'], reverse=True)
        return results

    def predict(self) -> Dict[str, Any]:
        """执行完整预测，返回带概率的详细结果"""
        result = self.fusion.predict(top_n=5, include_compound=True)
        groups = self.fusion.get_group_recommendations()
        back_recs = self.fusion.get_back_recommendations()
        bt = self.fusion.backtest(n_recent=100)

        # === 单式注 ===
        single_bets = []
        for bet in result['single_bets']:
            if self._is_recent_repeat(bet['front'], bet['back']):
                print(f"[优化器] ⛔ 过滤完全重复上期的前区: {bet['front']}")
                continue
            probs = self._calc_probability(bet['front'], bet['back'])
            single_bets.append({
                'front': bet['front'],
                'back': bet['back'],
                'strategy': bet.get('strategy_name', '融合策略'),
                'score': round(bet.get('final_score', 0.5), 4),
                'probability': probs['combined'],
                'front_prob': probs['front'],
                'back_prob': probs['back'],
                'math_score': probs['math_score'],
            })
        single_bets.sort(key=lambda x: x['score'], reverse=True)

        # === 各池策略胆拖方案 ===
        pool_dantuo = {}
        for pool_name, pool_info in self.POOL_STRATEGIES.items():
            dt_schemes = self._generate_dan_tuo(pool_name)
            pool_dantuo[pool_name] = {
                'name': pool_info['name'],
                'desc': pool_info['desc'],
                'schemes': dt_schemes[:4],  # top 4
            }

        # === 系统复式（保持原样，额外补充） ===
        compound_bets = result.get('compound_bets', [])
        compound_list = []
        if isinstance(compound_bets, dict):
            for bt_name, bets in compound_bets.items():
                if isinstance(bets, list) and bets:
                    for bet in bets[:1]:
                        if isinstance(bet, dict):
                            probs = self._calc_probability(
                                bet.get('front', [1,2,3,4,5]),
                                bet.get('back', [1,2])
                            )
                            compound_list.append({
                                'bet_type': bt_name,
                                'front': bet.get('front', []),
                                'back': bet.get('back', []),
                                'strategy': bet.get('pool_type', 'mixed'),
                                'strategy_score': bet.get('strategy_score', 0),
                                'probability': probs['combined'],
                            })

        return {
            'single_bets': single_bets,
            'pool_dantuo': pool_dantuo,
            'compound_bets': compound_list,
            'backtest': bt,
        }


def _prob_bar(pct: float, width: int = 15) -> str:
    """概率条可视化"""
    filled = int(pct / 100 * width)
    filled = min(filled, width)
    return '█' * filled + '░' * (width - filled)


def format_prediction(result: Dict[str, Any]) -> str:
    """格式化为人类可读的预测报告"""
    lines = []
    lines.append("🏆 大乐透 预测报告")
    lines.append("=" * 65)
    lines.append("")

    # =====================
    # 单式投注
    # =====================
    lines.append("🎯 【单式投注 Top 5】")
    lines.append("-" * 65)
    for i, bet in enumerate(result['single_bets'], 1):
        f_str = '  '.join(f"{n:>2}" for n in bet['front'])
        b_str = '  '.join(f"{n:>2}" for n in bet['back'])
        bar = _prob_bar(bet['probability'])
        lines.append(f"  第{i}注: [{f_str}] + [{b_str}]")
        lines.append(f"        🎲 命中概率: {bet['probability']:.1f}%  {bar}")
        lines.append(f"        前区={bet['front_prob']:.1f}%  后区={bet['back_prob']:.1f}%  数学约束分={bet['math_score']:.1f}%")
        lines.append(f"        策略: {bet['strategy']}  |  综合评分: {bet['score']:.4f}")
        lines.append("")

    # =====================
    # 各池策略胆拖方案
    # =====================
    lines.append("📊 【多池策略 · 胆拖投注方案】")
    lines.append("-" * 65)

    for pool_name, pool_data in result['pool_dantuo'].items():
        name = pool_data['name']
        desc = pool_data['desc']
        schemes = pool_data['schemes']

        lines.append(f"  {name}（{desc}）")

        if not schemes:
            lines.append(f"    └─ 暂无有效推荐")
            lines.append("")
            continue

        for idx, s in enumerate(schemes):
            prefix = "┌─" if idx == 0 else "├─"
            bar = _prob_bar(s['probability'])
            back_str = ', '.join(map(str, s['back']))

            # 如果胆拖只有1注 => 展开为完整号码展示
            if s['total_bets'] == 1:
                full_front = sorted(s['front_dan'] + s['front_tuo'])
                f_str = '  '.join(f"{n:>2}" for n in full_front)
                lines.append(f"    {prefix} [{f_str}] + [{back_str}]")
                lines.append(f"    │   🎲 命中概率: {s['probability']:.1f}%  {bar}")
                lines.append(f"    │   前区概率: {s['front_prob']:.1f}%  后区概率: {s['back_prob']:.1f}%")
            else:
                dan_str = ', '.join(map(str, s['front_dan']))
                tuo_str = ', '.join(map(str, s['front_tuo']))
                lines.append(f"    {prefix} {s['name']}  (共{s['total_bets']}注)")
                lines.append(f"    │   前区胆[{dan_str}]  拖[{tuo_str}]")
                lines.append(f"    │   后区[{back_str}] ({s['back_desc']})")
                lines.append(f"    │   🎲 命中概率: {s['probability']:.1f}%  {bar}")
                lines.append(f"    │   前区概率: {s['front_prob']:.1f}%  后区概率: {s['back_prob']:.1f}%")

            if idx == 0:
                lines.append(f"    │")
                lines.append(f"    ├─ 备选方案(同策略其他方案):")

        lines.append("")

    # =====================
    # 系统复式（作为补充）
    # =====================
    lines.append("🔗 【系统推荐复式投注（备选）】")
    lines.append("-" * 65)
    if result['compound_bets']:
        for i, cb in enumerate(result['compound_bets'], 1):
            f_str = ', '.join(map(str, cb['front']))
            b_str = ', '.join(map(str, cb['back']))
            bar = _prob_bar(cb['probability'])
            lines.append(f"  {cb['bet_type']:<6}: [{f_str}] + [{b_str}]")
            lines.append(f"          🎲 命中概率: {cb['probability']:.1f}%  {bar}")
            lines.append(f"          策略={cb['strategy']}  评分={cb['strategy_score']}")
            lines.append("")
    else:
        lines.append("  (无)")

    # =====================
    # 回测参考
    # =====================
    bt = result.get('backtest', {})
    if bt and 'error' not in bt:
        lines.append("📈 【回测参考（最近100期）】")
        lines.append("-" * 65)
        rand_front = bt.get('random_baseline', {}).get('front_per_draw', 0.714)
        rand_back = bt.get('random_baseline', {}).get('back_per_draw', 0.333)
        lines.append(f"  随机基准: 前区 {rand_front:.3f}/注  后区 {rand_back:.3f}/注")
        lines.append("")
        # 策略命中概率汇总表
        pool_names_display = {
            'hot': '🔥 热号', 'cold': '❄️ 冷号', 'balance': '⚖️ 均衡',
            'game_theory': '🎯 博弈', 'genetic': '🧬 遗传'
        }
        perfs = bt.get('pool_performance', {})
        imps = bt.get('improvement_vs_random', {})
        lines.append(f"  {'策略':<10} {'前区均值':>12} {'前区提升':>10} {'后区均值':>12} {'后区提升':>10}")
        lines.append(f"  " + "-" * 56)
        for pname, perf in perfs.items():
            imp = imps.get(pname, {})
            pname_display = pool_names_display.get(pname, pname.capitalize())
            front_imp = imp.get('front_improvement_%', 0)
            back_imp = imp.get('back_improvement_%', 0)
            # 转为星级表示
            front_star = '⭐' * min(int(front_imp / 80), 3) if front_imp > 0 else ''
            back_star = '⭐' * min(int(back_imp / 80), 3) if back_imp > 0 else ''
            lines.append(f"  {pname_display:<10} {perf['avg_front_hits']:.3f}/5 ({front_imp:+.0f}%){front_star:<4} "
                       f"{perf['avg_back_hits']:.3f}/2 ({back_imp:+.0f}%){back_star:<4}")
    lines.append("")
    lines.append("=" * 65)
    lines.append("⚠️ 彩票有风险，以上预测仅供参考")
    return '\n'.join(lines)


def main():
    print("🔮 DLT大乐透优化预测器 V1.0 (胆拖版)")
    print("=" * 65)
    predictor = DLTOptimizedPredictor()
    result = predictor.predict()
    print(format_prediction(result))

if __name__ == '__main__':
    main()
