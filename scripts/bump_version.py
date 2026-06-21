#!/usr/bin/env python3
"""
DLT 版本升级工具 — 一次命令更新所有版本号

用法：
    python3 bump_version.py              # 查看当前版本
    python3 bump_version.py 3.2.0        # 升级到 3.2.0
    python3 bump_version.py 3.2.0 2026-06-20  # 指定日期

自动更新：
  - scripts/version.py          ← 单源版本定义
  - references/dlt_skill_config.json ← skill_name / version / reference_sync_version
  - SKILL.md                    ← 标题中的版本号
"""

import sys
import os
import re
import json

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SKILL_DIR)  # skills/dlt-lottery-prediction/
VERSION_FILE = os.path.join(SKILL_DIR, 'version.py')
CONFIG_FILE = os.path.join(REPO_DIR, 'references', 'dlt_skill_config.json')
SKILL_MD = os.path.join(REPO_DIR, 'SKILL.md')
FILES_AFFECTED = []


def read_version_from_file():
    """从 version.py 读取当前版本"""
    with open(VERSION_FILE) as f:
        for line in f:
            m = re.match(r'^VERSION\s*=\s*"(.+)"', line)
            if m:
                return m.group(1)
    return None


def update_version_file(new_ver, new_date):
    """更新 version.py"""
    with open(VERSION_FILE) as f:
        content = f.read()
    content = re.sub(r'^VERSION\s*=.*', f'VERSION = "{new_ver}"', content, flags=re.MULTILINE)
    content = re.sub(r'^RELEASE_DATE\s*=.*', f'RELEASE_DATE = "{new_date}"', content, flags=re.MULTILINE)
    with open(VERSION_FILE, 'w') as f:
        f.write(content)
    FILES_AFFECTED.append('version.py')


def update_config_json(new_ver):
    """更新 dlt_skill_config.json 中的版本字段"""
    with open(CONFIG_FILE) as f:
        config = json.load(f)

    updated = False
    for key in ['skill_name', 'version', 'reference_sync_version']:
        if key in config:
            old_val = config[key]
            if key == 'skill_name':
                # "大乐透预测 V3.1.2" → "大乐透预测 V3.1.3"
                new_val = re.sub(r'V[\d.]+', f'V{new_ver}', old_val)
            else:
                new_val = new_ver
            if old_val != new_val:
                config[key] = new_val
                updated = True

    if updated:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            f.write('\n')
        FILES_AFFECTED.append('references/dlt_skill_config.json')


def update_skill_md(new_ver):
    """更新 SKILL.md 标题中的版本号"""
    with open(SKILL_MD) as f:
        content = f.read()
    new_content = re.sub(r'# DLT大乐透预测技能 V[\d.]+', f'# DLT大乐透预测技能 V{new_ver}', content)
    if new_content != content:
        with open(SKILL_MD, 'w') as f:
            f.write(new_content)
        FILES_AFFECTED.append('SKILL.md')


def print_summary(new_ver, new_date):
    print("=" * 50)
    print(f"  DLT 版本升级完成")
    print(f"  {old_ver} → {new_ver} ({new_date})")
    print("=" * 50)
    print(f"  更新文件 ({len(FILES_AFFECTED)} 个):")
    for f in FILES_AFFECTED:
        print(f"    ✓ {f}")
    print()
    print(f"  ✅ 运行 check_reference_sync() 确认同步正常")


if __name__ == '__main__':
    # 读取当前版本
    old_ver = read_version_from_file()
    if not old_ver:
        print("❌ 无法读取 version.py，请检查文件格式")
        sys.exit(1)

    if len(sys.argv) < 2:
        print(f"当前版本: {old_ver}")
        print(f"用法: python3 {os.path.basename(__file__)} <版本号> [日期]")
        print(f"示例: python3 {os.path.basename(__file__)} 3.2.0")
        print(f"      python3 {os.path.basename(__file__)} 3.2.0 2026-06-20")
        sys.exit(0)

    new_ver = sys.argv[1]
    if not re.match(r'^\d+\.\d+\.\d+$', new_ver):
        print(f"❌ 版本号格式错误: {new_ver}，应为 X.Y.Z")
        sys.exit(1)

    new_date = sys.argv[2] if len(sys.argv) > 2 else None

    if new_date:
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', new_date):
            print(f"❌ 日期格式错误: {new_date}，应为 YYYY-MM-DD")
            sys.exit(1)
    else:
        from datetime import date
        new_date = date.today().isoformat()

    if old_ver == new_ver:
        print(f"当前已是 V{old_ver}，无需更新")
        sys.exit(0)

    update_version_file(new_ver, new_date)
    update_config_json(new_ver)
    update_skill_md(new_ver)

    print_summary(new_ver, new_date)
