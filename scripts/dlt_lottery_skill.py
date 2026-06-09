#!/usr/bin/env python3
"""
DLT彩票预测技能 — OpenClaw CLI Adapter
委托给 DLTFusionComplete 执行实际预测，本层只做参数解析+格式化输出。
"""

import argparse
import sys
import json
from typing import Dict, List, Any, Optional

from dlt_fusion_complete import DLTFusionComplete, data_dir


# ── 格式化工具 ─────────────────────────────────────────

def _fmt_front(nums: List[int]) -> str:
    return ' '.join(f'{n:02d}' for n in sorted(nums))


def _fmt_back(nums: List[int]) -> str:
    return ' '.join(f'{n:02d}' for n in sorted(nums))


def _print_predict(result: Dict[str, Any]) -> None:
    """格式化预测输出"""
    single = result.get('single_bets', [])
    compound = result.get('compound_bets', {})
    dan_tuo = result.get('dan_tuo_bets', {})
    period = result.get('period', '?')

    print(f"\n{'='*60}")
    print(f"  DLT 大乐透 第{period}期 多策略融合预测")
    print(f"{'='*60}")

    # 单式推荐
    print(f"\n🎯 单式方案 (Top {len(single)})")
    print(f"{'-'*50}")
    for i, bet in enumerate(single, 1):
        front = _fmt_front(bet.get('front', []))
        back = _fmt_back(bet.get('back', []))
        score = bet.get('final_score', 0)
        prob = bet.get('hit_probability', 0)
        print(f"  {i:2d}. [{front}] + [{back}]  score={score:.4f}  p={prob:.1f}%")

    # 复式方案
    if compound:
        print(f"\n📋 复式方案")
        print(f"{'-'*50}")
        for ctype, bets in compound.items():
            if not bets:
                continue
            n_bets = len(bets)
            front_cnt = bets[0].get('front_count', 6)
            back_cnt = bets[0].get('back_count', 3)
            from math import comb
            total_notes = comb(front_cnt, 5) * comb(back_cnt, 2)
            print(f"  {ctype} ({total_notes}注, {total_notes*2}元) × {n_bets}组:")
            for i, b in enumerate(bets, 1):
                # 优先使用完整候选池(front_pool/back_pool)，兼容旧格式(front/back)
                front_pool = b.get('front_pool', b.get('front', []))
                back_pool = b.get('back_pool', b.get('back', []))
                front = _fmt_front(front_pool)
                back = _fmt_back(back_pool)
                print(f"    {i}: [{front}] + [{back}]")

    # 胆拖方案
    if dan_tuo:
        print(f"\n📊 胆拖方案")
        print(f"{'-'*50}")
        for pn, info in dan_tuo.items():
            label = info.get('name', pn)
            schemes = info.get('schemes', [])
            print(f"  {label}:")
            for s in schemes:
                dan = _fmt_front(s.get('front_dan', []))
                # 优先使用完整拖池，兼容旧格式
                tuo = _fmt_front(s.get('front_tuo_pool', s.get('front_tuo', [])))
                back = _fmt_back(s.get('back', []))
                bets = s.get('total_bets', 0)
                prob = s.get('hit_probability', 0)
                name = s.get('name', '')
                print(f"    {name}: 胆[{dan}] 拖[{tuo}] 后区[{back}]  "
                      f"{bets}注{bets*2}元  p={prob:.1f}%")

    print(f"\n⚠️  仅供参考娱乐，请理性投注！")


def _print_backtest(result: Dict[str, Any]) -> None:
    """格式化回测输出"""
    if 'error' in result:
        print(f"❌ 回测失败: {result['error']}")
        return

    pool_perf = result.get('pool_performance', {})
    compound_cov = result.get('compound_coverage', {})

    print(f"\n{'='*60}")
    print(f"  DLT 回测报告")
    print(f"{'='*60}")

    print(f"\n📈 池级别命中率")
    print(f"{'─'*50}")
    for pool, perf in sorted(pool_perf.items()):
        fh = perf.get('front_hit_rate', 0)
        bh = perf.get('back_hit_rate', 0)
        rnd_f = perf.get('random_front_baseline', 0.714)
        rnd_b = perf.get('random_back_baseline', 0.333)
        print(f"  {pool:12s}: 前区{fh:.1f}(随机{rnd_f:.1f})  后区{bh:.1f}(随机{rnd_b:.1f})")

    if compound_cov:
        print(f"\n📋 复式覆盖率")
        print(f"{'─'*50}")
        for ctype, cov in sorted(compound_cov.items()):
            print(f"  {ctype}: 覆盖率{cov*100:.1f}%")

    summary = result.get('summary', {})
    if summary:
        print(f"\n📊 汇总")
        print(f"{'─'*50}")
        print(f"  回测期数: {summary.get('test_periods', '?')}")
        print(f"  模型提升(前区): +{summary.get('front_improvement', 0)*100:.1f}%")
        print(f"  模型提升(后区): +{summary.get('back_improvement', 0)*100:.1f}%")


def _print_compound(bets: Dict[str, List[Dict]]) -> None:
    """格式化复式输出"""
    print(f"\n{'='*60}")
    print(f"  复式投注方案")
    print(f"{'='*60}")

    total_cost = 0
    for ctype, group in sorted(bets.items()):
        if not group:
            continue
        b = group[0]
        fc, bc = b.get('front_count', 6), b.get('back_count', 3)
        from math import comb
        notes = comb(fc, 5) * comb(bc, 2)
        cost = notes * 2
        total_cost += cost * len(group)
        print(f"\n  {ctype} ({notes}注, {cost}元) × {len(group)}组:")
        for i, bet in enumerate(group, 1):
            front = _fmt_front(bet.get('front', []))
            back = _fmt_back(bet.get('back', []))
            print(f"    {i}. [{front}] + [{back}]")

    print(f"\n💰 合计: {total_cost}元")
    print(f"\n⚠️  仅供参考娱乐，请理性投注！")


def _print_dantuo(results: List[Dict]) -> None:
    """格式化胆拖输出"""
    print(f"\n{'='*60}")
    print(f"  胆拖投注方案")
    print(f"{'='*60}")

    for i, r in enumerate(results, 1):
        dan = _fmt_front(r.get('front_dan', []))
        tuo = _fmt_front(r.get('front_tuo', []))
        back = _fmt_back(r.get('back', []))
        name = r.get('name', '')
        bets = r.get('total_bets', 0)
        prob = r.get('hit_probability', 0)
        print(f"\n  {i}. {name}")
        print(f"     胆码: [{dan}]")
        print(f"     拖码: [{tuo}]")
        print(f"     后区: [{back}]")
        print(f"     {bets}注 {bets*2}元  p={prob:.1f}%")


def _print_stake(stakes: List[Dict], budget: float) -> None:
    """格式化凯利投注建议"""
    print(f"\n{'='*60}")
    print(f"  凯利公式投注建议 (预算: {budget}元)")
    print(f"{'='*60}")
    print(f"{'注':4s} {'前区':18s} {'后区':8s} {'命中率':8s} {'凯利比':8s} {'建议金额':10s}")
    print(f"{'─'*56}")
    total = 0
    for i, s in enumerate(stakes, 1):
        front = _fmt_front(s.get('front', []))
        back = _fmt_back(s.get('back', []))
        hp = s.get('hit_prob', 0)
        kp = s.get('kelly_pct', 0)
        stake = s.get('stake_yuan', 0)
        total += stake
        note = s.get('note', '')
        print(f"  {i:2d}  [{front}] [{back}] {hp:>5.1f}%  {kp:>6.4f}%  {stake:>7.2f}元  {note}")
    print(f"{'─'*56}")
    print(f"  {'合计':>44s} {total:.2f}元")


# ── CLI ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DLT大乐透预测技能 — 基于多策略融合+神经网络集成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  predict                   预测下一期号码
  predict --top-k 3         生成3注单式
  predict --no-compound     只生成单式，跳过复式
  compound                  生成全部复式方案
  compound --type 7+3       仅生成7+3复式
  dantuo                    自动胆拖投注
  dantuo --dan-front 8 22   指定前区胆码
  backtest --periods 50     回测50期
  stake --budget 200        凯利公式投注建议（预算200元）
  info                      查看技能信息
        """,
    )
    parser.add_argument('command', nargs='?', default='predict',
                        choices=['predict', 'compound', 'dantuo', 'backtest',
                                 'stake', 'info'],
                        help='技能命令')

    # predict
    parser.add_argument('--top-k', type=int, default=5,
                        help='预测方案数量 (默认5)')
    parser.add_argument('--no-compound', action='store_true',
                        help='不生成复式方案')

    # compound
    parser.add_argument('--type', type=str, default='all',
                        help='复式类型: 6+3,7+2,8+3,9+4,all (默认all)')
    parser.add_argument('--n-per-type', type=int, default=2,
                        help='每种类型生成组数 (默认2)')

    # dantuo
    parser.add_argument('--dan-front', type=int, nargs='*', default=None,
                        help='前区胆码 (1-4个, 默认自动选)')
    parser.add_argument('--tuo-size', type=int, default=8,
                        help='拖码池大小 (默认8)')
    parser.add_argument('--dan-back', type=int, nargs='*', default=None,
                        help='后区胆码 (0-1个, 默认自动选)')
    parser.add_argument('--n-sets', type=int, default=3,
                        help='胆拖输出组数 (默认3)')

    # backtest
    parser.add_argument('--periods', type=int, default=100,
                        help='回测期数 (默认100)')

    # stake
    parser.add_argument('--budget', type=float, default=100.0,
                        help='预算金额,元 (默认100)')
    parser.add_argument('--half-kelly', action='store_true', default=True,
                        help='使用half-Kelly (默认开启)')
    parser.add_argument('--no-half-kelly', dest='half_kelly', action='store_false',
                        help='使用full Kelly')

    args = parser.parse_args()

    # 初始化
    try:
        dp = data_dir()
        fusion = DLTFusionComplete(dp)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)

    # 分发命令
    try:
        if args.command == 'predict':
            result = fusion.predict(top_n=args.top_k,
                                    include_compound=not args.no_compound)
            _print_predict(result)

        elif args.command == 'compound':
            if args.type == 'all':
                bets = fusion.generate_compound_bets('all',
                                                      n_per_type=args.n_per_type)
            else:
                bets = fusion.generate_compound_bets(args.type,
                                                      n_per_type=args.n_per_type)
            _print_compound(bets)

        elif args.command == 'dantuo':
            results = fusion.generate_dantuo_bets(
                dan_front=args.dan_front,
                tuo_front_size=args.tuo_size,
                dan_back=args.dan_back,
                n_sets=args.n_sets,
            )
            if results:
                _print_dantuo(results)
            else:
                print("❌ 胆拖投注生成失败（候选池可能不足）")

        elif args.command == 'backtest':
            result = fusion.backtest(n_recent=args.periods)
            _print_backtest(result)

        elif args.command == 'stake':
            # 先预测，再计算凯利
            pred = fusion.predict(top_n=5, include_compound=False)
            singles = pred.get('single_bets', [])
            stakes = fusion.recommend_stake(
                budget=args.budget,
                predictions=singles,
                half_kelly=args.half_kelly,
            )
            if stakes:
                _print_stake(stakes, args.budget)
            else:
                print("❌ 没有可投注的方案")

        elif args.command == 'info':
            print(f"\n{'='*60}")
            print(f"  DLT 大乐透多策略融合预测系统")
            print(f"  V3.0.0 — 六池采样+跨期模式识别+NeuralEnsemble")
            print(f"{'='*60}")
            print(f"\n核心策略:")
            print(f"  🔥 热号池(30%)  ❄️ 冷号池(15%)  ⚖️ 均衡池(20%)")
            print(f"  📈 趋势池(20%)  🎯 博弈池(10%)  🧬 遗传池(5%)")
            print(f"\n跨期模式识别: 跨度/连号/和值/奇偶比/重号/尾号/三区/质数/AC值")
            print(f"\n🧠 神经网络: TabNet(25%) + LSTM(35%) + Transformer(40%)")
            print(f"\n复式方案: 12种 (6+3 ~ 9+6)")
            print(f"胆拖方案: 支持自定义胆码+自动胆码")
            print(f"凯利公式: half-Kelly 资金管理")
            print(f"\n数据: 自动同步体彩API + 500彩票网双源fallback")
            print(f"\n技巧: dlt-lottery-prediction --help 查看所有命令")

    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
