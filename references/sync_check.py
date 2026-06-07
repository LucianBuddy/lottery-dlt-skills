#!/usr/bin/env python3
"""
references/ 版本同步检查工具
使用方式: python sync_check.py [--fix]

功能：
1. 读取 dlt_fusion_complete.py 中的 VERSION 常量
2. 读取 references/ 下各文件中的版本标记
3. 对比一致性，输出差异报告
4. --fix 模式：自动更新过时文件的版本号（需要仔细审查后再提交）
"""

import sys
import os
import json
import re

# 定位技能目录
SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(SKILL_DIR, 'scripts')
REFERENCES_DIR = os.path.join(SKILL_DIR, 'references')


def get_code_version() -> str:
    """从 dlt_fusion_complete.py 读取版本号"""
    path = os.path.join(SCRIPTS_DIR, 'dlt_fusion_complete.py')
    with open(path, 'r') as f:
        content = f.read()
    match = re.search(r'^VERSION\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if match:
        return match.group(1)
    return "unknown"


def check_json_config(source: str, path: str, code_ver: str) -> dict:
    """检查 JSON 配置文件的版本一致性"""
    result = {'file': path, 'status': 'UNKNOWN', 'stored_version': None}
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        ref_ver = data.get('reference_sync_version') or data.get('version', '')
        result['stored_version'] = str(ref_ver)
        if 'reference_sync_version' in data:
            result['matched'] = (str(ref_ver) == code_ver)
            result['status'] = 'OK' if result['matched'] else 'MISMATCH'
        else:
            result['matched'] = (str(ref_ver) == code_ver)
            result['status'] = 'OK (via version field)' if result['matched'] else 'MISMATCH (via version field)'
    except Exception as e:
        result['status'] = f'ERROR: {e}'
    return result


def check_yaml(source: str, path: str, code_ver: str) -> dict:
    """检查 YAML 文件的版本一致性（简单文本解析）"""
    result = {'file': path, 'status': 'UNKNOWN', 'stored_version': None}
    try:
        with open(path, 'r') as f:
            content = f.read()
        ver_match = re.search(r'reference_sync_version:\s*["\']?([^"\'\n]+)["\']?', content)
        if ver_match:
            result['stored_version'] = ver_match.group(1).strip()
            result['matched'] = (result['stored_version'] == code_ver)
            result['status'] = 'OK' if result['matched'] else 'MISMATCH'
        else:
            # fallback: check version field
            ver_match = re.search(r'^version:\s*["\']?([^"\'\n]+)["\']?', content, re.MULTILINE)
            if ver_match:
                result['stored_version'] = ver_match.group(1).strip()
                result['matched'] = (result['stored_version'] == code_ver)
                result['status'] = 'OK (via version field)' if result['matched'] else 'MISMATCH (via version field)'
            else:
                result['status'] = 'NO VERSION MARKER'
    except Exception as e:
        result['status'] = f'ERROR: {e}'
    return result


def check_markdown(source: str, path: str, code_ver: str) -> dict:
    """检查 Markdown 文件的版本标记"""
    result = {'file': path, 'status': 'UNKNOWN', 'stored_version': None}
    try:
        with open(path, 'r') as f:
            content = f.read()
        # 查找 "当前版本：**Vxxx**" 或 "版本：Vxxx" 等标记
        patterns = [
            r'版本[：:]\s*\*{0,2}V?([\d.]+)\*{0,2}',
            r'Version[：:]\s*\*{0,2}V?([\d.]+)\*{0,2}',
        ]
        for pat in patterns:
            match = re.search(pat, content)
            if match:
                result['stored_version'] = match.group(1).strip()
                break
        if result['stored_version']:
            result['matched'] = (result['stored_version'] == code_ver)
            result['status'] = 'OK' if result['matched'] else 'MISMATCH'
        else:
            result['status'] = 'NO VERSION MARKER'
    except Exception as e:
        result['status'] = f'ERROR: {e}'
    return result


def main():
    code_ver = get_code_version()
    print(f"当前代码版本: V{code_ver}")
    print(f"{'='*50}")

    # 列出 references/ 下所有文件
    files_to_check = []
    for fname in os.listdir(REFERENCES_DIR):
        fpath = os.path.join(REFERENCES_DIR, fname)
        if os.path.isfile(fpath):
            files_to_check.append(fpath)

    # 检查策略
    checkers = {
        '.json': check_json_config,
        '.yaml': check_yaml,
        '.yml': check_yaml,
        '.md': check_markdown,
    }

    all_ok = True
    results = []
    for fpath in sorted(files_to_check):
        ext = os.path.splitext(fpath)[1].lower()
        checker = checkers.get(ext)
        if not checker:
            results.append({'file': fpath, 'status': f'SKIPPED (unsupported: {ext})'})
            continue
        r = checker('code', fpath, code_ver)
        results.append(r)

    for r in results:
        basename = os.path.basename(r['file'])
        if r['status'].startswith('OK'):
            print(f"  ✅ {basename}: {r['status']} (v{r['stored_version']})")
        elif r['status'].startswith('MISMATCH'):
            print(f"  ❌ {basename}: 版本不匹配! 代码 V{code_ver} ≠ 文件 V{r['stored_version']}")
            all_ok = False
        elif r['status'].startswith('NO VERSION'):
            print(f"  ⚠️  {basename}: 未找到版本标记")
        elif r['status'].startswith('ERROR'):
            print(f"  ❌ {basename}: {r['status']}")
            all_ok = False
        else:
            print(f"  ❓ {basename}: {r['status']}")

    print(f"\n{'='*50}")
    if all_ok:
        print("✅ 所有 references 文件版本同步正常")
    else:
        print(f"❌ 存在版本不同步的文件，请更新 references/ 下的对应文件")
        sys.exit(1)


if __name__ == '__main__':
    main()
