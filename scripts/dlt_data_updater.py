#!/usr/bin/env python3
"""
DLT大乐透历史数据自动更新器 v2
数据源: 体彩数据API (webapi.sporttery.cn)
注意: 该 API 受 EdgeOne 安全防护，部分环境可能被拦截
"""

import os
import sys
import json
import re
import urllib.request
import ssl
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

SKILL_DIR = Path(__file__).resolve().parent
DATA_PATH = SKILL_DIR.parent / 'assets' / 'data' / 'DLT历史数据_适配模型版.xlsx'

# 体彩数据API (sporttery.cn)
# gameNo=85 是超级大乐透
API_URL = 'https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry'

# 500彩票网数据源（备选）
API_URL_500COM = 'https://datachart.500.com/dlt/history/newinc/history.php'


def _build_url(page_no: int = 1, page_size: int = 30) -> str:
    """构建体彩数据API请求URL"""
    return (f'{API_URL}?gameNo=85&provinceId=0&pageSize={page_size}'
            f'&isPc=true&pageNo={page_no}')


def _http_get(url: str, timeout: int = 15, referer: str = 'https://www.lottery.gov.cn/') -> str:
    """HTTP GET请求（带SSL绕过和浏览器UA）"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url, headers={
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'application/json, text/plain, */*',
        'Referer': referer,
    })

    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return resp.read().decode('utf-8', errors='replace')


def fetch_draws_from_api(page_no: int = 1, page_size: int = 30) -> List[dict]:
    """
    从体彩数据API获取大乐透历史开奖数据

    Returns:
        [{'期号': int, '前区1'~'前区5': int, '后区1': int, '后区2': int}, ...]
    """
    url = _build_url(page_no, page_size)
    try:
        raw = _http_get(url)
        data = json.loads(raw)

        if not data.get('success'):
            print(f"[DLT-Updater] API返回失败: {data.get('errorMessage', '未知错误')}")
            return []

        draw_list = data.get('value', {}).get('list', [])
        results = []
        for draw in draw_list:
            draw_num = draw.get('lotteryDrawNum', '')
            draw_result = draw.get('lotteryDrawResult', '')
            draw_time = draw.get('lotteryDrawTime', '')

            if not draw_num or not draw_result:
                continue

            # 解析号码："02 03 20 28 33 02 12"
            nums = draw_result.strip().split()
            if len(nums) != 7:
                continue

            try:
                period = int(draw_num)
                results.append({
                    '期号': period,
                    '前区1': int(nums[0]), '前区2': int(nums[1]),
                    '前区3': int(nums[2]), '前区4': int(nums[3]),
                    '前区5': int(nums[4]),
                    '后区1': int(nums[5]), '后区2': int(nums[6]),
                    # 保留开奖日期用于日志
                    '_date': draw_time,
                })
            except (ValueError, IndexError):
                continue

        return results

    except Exception as e:
        print(f"[DLT-Updater] API请求失败: {e}")
        return []


# ——— 500彩票网数据源（备选） ———

def _http_get_500com(timeout: int = 20) -> str:
    """500.com专用HTTP请求（不同referer和更长的超时）"""
    url = f'{API_URL_500COM}?start=7001&end=99999'
    return _http_get(url, timeout=timeout, referer='https://www.500.com/')


def parse_500com_html(html: str) -> List[dict]:
    """
    解析500彩票网HTML表格，提取大乐透开奖数据

    HTML结构（每行首列被注释隐藏）:
    <tr class="t_tr1">
        <!--<td>序号</td>--><td class="t_tr1">期号</td>
        <td class="cfont2">前区1</td> ... <td class="cfont2">前区5</td>
        <td class="cfont4">后区1</td>
        <td class="cfont4">后区2</td>
        ...
    </tr>
    """
    # 先去掉HTML注释，避免干扰
    clean = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

    results = []
    # 匹配每个数据行
    row_pattern = re.compile(
        r'<tr\s+class="t_tr1">(.*?)</tr>', re.DOTALL
    )
    # 提取所有td中的纯数字
    td_pattern = re.compile(r'<td[^>]*>(\d+)</td>')

    for row in row_pattern.finditer(clean):
        cells = td_pattern.findall(row.group(1))
        # 需要至少7个有效数字列：期号 + 前区5 + 后区2
        if len(cells) < 7:
            continue

        try:
            period = int(cells[0])
            front = [int(cells[i]) for i in range(1, 6)]
            back = [int(cells[6]), int(cells[7])]

            # 验证范围
            if not all(1 <= n <= 35 for n in front):
                continue
            if not all(1 <= n <= 12 for n in back):
                continue

            results.append({
                '期号': period,
                '前区1': front[0], '前区2': front[1],
                '前区3': front[2], '前区4': front[3],
                '前区5': front[4],
                '后区1': back[0], '后区2': back[1],
            })
        except (ValueError, IndexError):
            continue

    if results:
        # 按期号升序排列
        results.sort(key=lambda x: x['期号'])
        # 去重
        seen = set()
        unique = []
        for d in results:
            pid = d['期号']
            if pid not in seen:
                seen.add(pid)
                unique.append(d)
        results = unique

    return results


def fetch_draws_from_500com() -> List[dict]:
    """
    从500彩票网获取大乐透历史开奖数据（HTML表格解析）
    作为体彩数据API的备用数据源

    Returns:
        [{'期号': int, '前区1'~'前区5': int, '后区1': int, '后区2': int}, ...]
    """
    try:
        html = _http_get_500com()
        draws = parse_500com_html(html)
        if draws:
            print(f"[DLT-Updater] 500彩票网获取到 {len(draws)} 期数据 "
                  f"({draws[0]['期号']}~{draws[-1]['期号']})")
        else:
            print(f"[DLT-Updater] 500彩票网未解析到有效数据")
        return draws
    except Exception as e:
        print(f"[DLT-Updater] 500彩票网请求失败: {e}")
        return []


def fetch_new_500com(last_period: int) -> List[dict]:
    """
    从500彩票网获取比 last_period 更新的所有期号

    Returns:
        List[dict]: 新数据，按期号升序排列
    """
    all_draws = fetch_draws_from_500com()
    if not all_draws:
        return []

    # 过滤出新数据
    new_draws = [d for d in all_draws if d['期号'] > last_period]
    if new_draws:
        print(f"[DLT-Updater] 500彩票网发现 {len(new_draws)} 期新数据: "
              f"{new_draws[0]['期号']}~{new_draws[-1]['期号']}")
    else:
        current_max = max(d['期号'] for d in all_draws)
        print(f"[DLT-Updater] 数据已是最新 (最新期号: {current_max})")

    return new_draws


# ——— 主要获取逻辑（sporttery → 500com 双源） ———

def fetch_new_draws(last_period: int) -> List[dict]:
    """
    从体彩数据API获取比 last_period 更新的所有期号

    Returns:
        List[dict]: 新数据，按期号升序排列
    """
    all_draws = []
    page = 1
    max_pages = 5  # 最多查5页（150期），足够

    while page <= max_pages:
        draws = fetch_draws_from_api(page_no=page, page_size=30)
        if not draws:
            break

        all_draws.extend(draws)

        # 如果这一页已经全部 <= last_period，不再翻页
        if draws and draws[-1]['期号'] <= last_period:
            break

        page += 1

    if not all_draws:
        print(f"[DLT-Updater] 未获取到任何数据")
        return []

    # 去重（API可能重复）
    seen = set()
    unique = []
    for d in all_draws:
        pid = d['期号']
        if pid not in seen:
            seen.add(pid)
            unique.append(d)

    # 过滤出新数据
    new_draws = [d for d in unique if d['期号'] > last_period]
    new_draws.sort(key=lambda x: x['期号'])

    if new_draws:
        date_strs = [d.get('_date', '') for d in new_draws]
        print(f"[DLT-Updater] 体彩数据发现 {len(new_draws)} 期新数据: "
              f"{new_draws[0]['期号']}~{new_draws[-1]['期号']} "
              f"({date_strs[0]}~{date_strs[-1]})")
    else:
        current_max = max(d['期号'] for d in unique)
        print(f"[DLT-Updater] 数据已是最新 (最新期号: {current_max})")

    return new_draws


# ——— 文件读写 ———

def get_last_period() -> int:
    """读取Excel中最后一期的期号"""
    if not DATA_PATH.exists():
        return 0
    try:
        df = pd.read_excel(str(DATA_PATH), engine='openpyxl')
        return int(df['期号'].iloc[-1])
    except Exception as e:
        print(f"[DLT-Updater] 读取最后一期失败: {e}")
        return 0


def get_first_period() -> int:
    """读取Excel中第一期的期号"""
    if not DATA_PATH.exists():
        return 0
    try:
        df = pd.read_excel(str(DATA_PATH), engine='openpyxl')
        return int(df['期号'].iloc[0])
    except Exception:
        return 0


def append_draws(new_draws: List[dict]) -> int:
    """
    将新数据追加到Excel文件

    Returns:
        int: 追加的行数
    """
    if not new_draws:
        return 0

    # 去掉内部字段
    clean = [{k: v for k, v in d.items() if not k.startswith('_')} for d in new_draws]
    new_df = pd.DataFrame(clean)
    columns = ['期号', '前区1', '前区2', '前区3', '前区4', '前区5', '后区1', '后区2']

    for col in columns:
        if col not in new_df.columns:
            new_df[col] = 0
    new_df = new_df[columns]

    if DATA_PATH.exists():
        existing = pd.read_excel(str(DATA_PATH), engine='openpyxl')
        for col in columns:
            if col not in existing.columns:
                existing[col] = 0
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=['期号'], keep='last')
    combined = combined.sort_values('期号').reset_index(drop=True)
    combined.to_excel(str(DATA_PATH), index=False)

    print(f"[DLT-Updater] 数据文件已更新: {len(combined)} 期 "
          f"({combined['期号'].iloc[0]} ~ {combined['期号'].iloc[-1]})")
    return len(new_draws)


# ——— 核心接口 ———

def check_and_update() -> dict:
    """
    主入口：检查体彩数据API → 发现新数据 → 更新Excel

    Returns:
        dict:
            - updated: bool
            - new_count: int
            - last_period: int
            - first_period: int
            - total: int
            - new_periods: List[int]
            - source: str (标记数据来源)
    """
    last = get_last_period()
    first = get_first_period()

    print(f"[DLT-Updater] 当前数据文件: {first} ~ {last} ({get_total()}期)")

    # 优先使用体彩数据API
    new_draws = fetch_new_draws(last)
    source = 'sporttery'

    # 体彩失败则fallback到500彩票网
    if not new_draws:
        print(f"[DLT-Updater] ⚠️ 体彩数据API无数据，尝试500彩票网...")
        new_draws = fetch_new_500com(last)
        source = '500com'

    if not new_draws:
        return {
            'updated': False,
            'new_count': 0,
            'last_period': last,
            'first_period': first,
            'total': get_total(),
            'new_periods': [],
            'source': source,
        }

    count = append_draws(new_draws)
    new_periods = [d['期号'] for d in new_draws]

    return {
        'updated': True,
        'new_count': count,
        'last_period': new_periods[-1] if new_periods else last,
        'first_period': first,
        'total': get_total(),
        'new_periods': new_periods,
        'source': source,
    }


def get_total() -> int:
    """获取数据文件总期数"""
    if not DATA_PATH.exists():
        return 0
    try:
        df = pd.read_excel(str(DATA_PATH), engine='openpyxl')
        return len(df)
    except Exception:
        return 0


def verify_latest_period() -> dict:
    """
    仅检查最新一期（不写入文件），用于快速验证
    返回 {'period': int, 'date': str, 'result': str} 或 None
    """
    draws = fetch_draws_from_api(page_no=1, page_size=1)
    if draws:
        d = draws[0]
        return {
            'period': d['期号'],
            'date': d.get('_date', ''),
            'result': f"{d['前区1']:02d} {d['前区2']:02d} {d['前区3']:02d} "
                      f"{d['前区4']:02d} {d['前区5']:02d} + "
                      f"{d['后区1']:02d} {d['后区2']:02d}",
        }
    return {}


# ——— CLI ———

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DLT体彩数据更新器 v2')
    parser.add_argument('--check', action='store_true', help='仅检查最新一期')
    parser.add_argument('--latest', action='store_true', help='查看最新开奖')
    args = parser.parse_args()

    if args.latest:
        info = verify_latest_period()
        if info:
            print(f"最新期号: {info['period']}")
            print(f"开奖日期: {info['date']}")
            print(f"开奖号码: {info['result']}")
        else:
            print("未获取到数据")
    elif args.check:
        last = get_last_period()
        print(f"本地最新: {last}")
        new_draws = fetch_new_draws(last)
        if new_draws:
            print(f"有 {len(new_draws)} 期新数据:")
            for d in new_draws:
                print(f"  {d['期号']} ({d.get('_date','')}): "
                      f"{d['前区1']:02d} {d['前区2']:02d} {d['前区3']:02d} "
                      f"{d['前区4']:02d} {d['前区5']:02d} + "
                      f"{d['后区1']:02d} {d['后区2']:02d}")
        else:
            print("没有新数据")
    else:
        result = check_and_update()
        print(json.dumps(result, ensure_ascii=False, indent=2))
