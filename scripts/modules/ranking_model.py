#!/usr/bin/env python3
"""
DLT Ranking Model V1.0

基于sklearn GradientBoostingRegressor的可学习排序模型。
替代predict()中~20步串行评分pipeline，单次前向计算ranking score。

使用方法：
    model = DLTModel()
    model = model.train(draws, periods=None)   # 自动生成训练数据
    scores = model.predict(feature_vectors)    # 批量打分的

数据流：
    1. 对每期history N，用N-1之前的数据生成候选池
    2. 对每个候选提取feature + label = (实际中奖匹配数)
    3. 训练GBR模型预测match_count
    4. predict时用模型score替代final_score
"""

import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from modules.ranking_feature_extractor import extract_features, compute_match_label, compute_weighted_label

# sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, ndcg_score

# 模型文件路径
SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(SKILL_DIR, 'assets', 'data', 'ranking_model.pkl')


class DLTModel:
    """DLT可学习排序模型"""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        self.is_trained = False
        self._n_features = 66

    def predict_exclusion(self, features: List[List[float]]) -> List[float]:
        """【β】预测排除概率 — 号码在下一期不出现的可能性"""
        if not self.is_trained or self.model is None:
            return [0.5] * len(features)
        import numpy as np
        X = np.array(features, dtype=np.float32)
        if self.feature_mean is not None and self.feature_std is not None:
            X = (X - self.feature_mean) / (self.feature_std + 1e-8)
            X = np.clip(X, -5, 5)
        scores = self.model.predict(X)
        # 反转：GBR预测的是"出现概率"，排除概率 = 1 - 出现概率
        # 使用sigmoid转换确保在0-1范围
        excl = [1.0 / (1.0 + np.exp(-s)) for s in scores]
        return excl

    def predict(self, features: List[List[float]]) -> List[float]:

        """
        对特征向量列表批量预测评分。

        Args:
            features: shape=(n_candidates, n_features) 的特征列表

        Returns:
            scores: 每个候选的预测评分（高=更好）
        """
        if not self.is_trained or self.model is None:
            return None

        X = np.array(features, dtype=np.float32)
        # 标准化
        if self.feature_mean is not None and self.feature_std is not None:
            X = (X - self.feature_mean) / (self.feature_std + 1e-8)
            X = np.clip(X, -5, 5)

        scores = self.model.predict(X)
        return scores.tolist()

    def train(
        self,
        draws: List[Tuple[List[int], List[int]]],
        periods: Optional[List[int]] = None,
        n_range: int = 30,
        candidates_per_period: int = 50,
        force_retrain: bool = False,
        verbose: bool = True,
    ) -> 'DLTModel':
        """
        基于历史数据训练排序模型。

        策略：对每期，用之前的数据生成候选并打分，按实际开奖标注。

        Args:
            draws: 完整历史开奖数据
            periods: 期号列表
            n_range: 使用最近多少期生成训练数据
            candidates_per_period: 每期候选数
            force_retrain: 强制重训练
            verbose: 日志输出

        Returns:
            self（已训练）
        """
        if not force_retrain and self.is_trained:
            if verbose:
                print(f"[DLT-Model] 模型已训练，跳过")
            return self

        n_draws = len(draws)
        train_range = min(n_range, n_draws - 10)

        if train_range < 10:
            if verbose:
                print(f"[DLT-Model] ⚠️ 数据不足，跳过训练")
            return self

        all_features = []
        all_labels = []

        # 延迟导入以加速初始化
        from modules.dlt_pattern_recognizer import DLTPatternRecognizer

        if verbose:
            print(f"[DLT-Model] 🏋️ 开始训练数据生成 ({train_range}期)...")
            print(f"[DLT-Model]    每期{candidates_per_period}候选, "
                  f"共{train_range * candidates_per_period}样本")

        # 逐期生成训练数据
        for offset in range(1, train_range + 1):
            idx = n_draws - offset - 1  # 训练：用前idx的数据预测第idx+1期
            if idx < 10:
                continue

            train_draws = draws[:idx + 1]
            actual = draws[idx + 1]
            actual_front, actual_back = actual[0], actual[1]

            # 生成候选池
            candidates = self._generate_training_candidates(
                train_draws, candidates_per_period
            )

            # 提取特征+标签
            p_recognizer = None
            try:
                p_recognizer = DLTPatternRecognizer(train_draws)
                p_recognizer.build_distributions(window=min(500, len(train_draws)))
            except Exception:
                pass

            for cand in candidates:
                pattern_scores = None
                if p_recognizer and p_recognizer._is_built:
                    try:
                        ps = p_recognizer.score_combo(cand['front'], train_draws[-1][0] if len(train_draws) >= 1 else None)
                        pattern_scores = {'total_score': float(ps['total_score'])}
                    except Exception:
                        pass

                feat = extract_features(cand, train_draws, pattern_scores=pattern_scores)
                label = compute_weighted_label(cand['front'], cand['back'], actual_front, actual_back)
                all_features.append(feat)
                all_labels.append(label)

            if verbose and (offset % 10 == 0 or offset == train_range):
                print(f"[DLT-Model]   进度: {offset}/{train_range} 期, "
                      f"累计样本: {len(all_features)}")

        if len(all_features) < 100:
            if verbose:
                print(f"[DLT-Model] ⚠️ 样本不足({len(all_features)}), 跳过训练")
            return self

        # 转换为numpy
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.float32)

        # 标准化
        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.feature_mean) / self.feature_std
        X_norm = np.clip(X_norm, -5, 5)

        if verbose:
            print(f"[DLT-Model] 📊 训练数据: {X.shape} 样本×特征, "
                  f"标签范围: [{y.min():.2f}, {y.max():.2f}]")

        # 训练/验证
        X_train, X_val, y_train, y_val = train_test_split(
            X_norm, y, test_size=0.2, random_state=42
        )

        # GBR参数
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            min_samples_leaf=10,
            max_features='sqrt',
            loss='ls',
            random_state=42,
            verbose=0,
        )
        self.model.fit(X_train, y_train)

        # 验证
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)

        self.is_trained = True

        if verbose:
            print(f"[DLT-Model] ✅ 训练完成")
            print(f"[DLT-Model]    Train: R²={train_score:.4f}  MAE={train_mae:.4f}")
            print(f"[DLT-Model]    Val:   R²={val_score:.4f}  MAE={val_mae:.4f}")
            print(f"[DLT-Model]    特征重要性Top5:")
            importances = self.model.feature_importances_
            top5 = np.argsort(importances)[-5:][::-1]
            from modules.ranking_feature_extractor import FEATURE_NAMES
            for idx in top5:
                print(f"      [{idx}] {FEATURE_NAMES[idx]}: {importances[idx]:.4f}")

            # 保存模型
            self.save()
            print(f"[DLT-Model] 💾 模型已保存: {self.model_path}")

        return self

    def predict_single(self, features: List[float]) -> float:
        """单候选评分"""
        scores = self.predict([features])
        if scores and len(scores) > 0:
            return scores[0]
        return 0.0

    def save(self, path: Optional[str] = None) -> str:
        """保存模型到磁盘"""
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data = {
            'model': self.model,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'is_trained': self.is_trained,
            'n_features': self._n_features,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        return save_path

    def load(self, path: Optional[str] = None) -> bool:
        """从磁盘加载模型"""
        load_path = path or self.model_path
        if not os.path.exists(load_path):
            return False
        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.feature_mean = data['feature_mean']
            self.feature_std = data['feature_std']
            self.is_trained = data.get('is_trained', True)
            self._n_features = data.get("n_features", 66)
            return True
        except Exception as e:
            print(f"[DLT-Model] ⚠️ 加载失败: {e}")
            return False

    def _generate_training_candidates(
        self, draws: List[Tuple[List[int], List[int]]],
        n_candidates: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        为训练生成候选池。
        使用简单的多池采样 + 随机扰动，避免循环依赖。
        """
        from collections import Counter
        candidates = []
        n_draws = len(draws)
        if n_draws < 20:
            return candidates

        window = min(50, n_draws)

        # 热号池
        recent_counter = Counter()
        for d in draws[-window:]:
            recent_counter.update(d[0])
        hot_pool = [n for n, _ in recent_counter.most_common(12)]

        # 冷号池
        all_hot_set = set(hot_pool)
        cold_pool = [n for n in range(1, 36) if n not in all_hot_set][:8]

        # 均衡池（范围限制）
        balanced = []
        zones = [list(range(1, 13)), list(range(13, 25)), list(range(25, 36))]
        for z in zones:
            z_nums = [n for n in z if n not in all_hot_set]
            if len(z_nums) >= 2:
                balanced.extend(np.random.choice(z_nums, min(3, len(z_nums)), replace=False).tolist())
            else:
                balanced.extend(np.random.choice(z, min(3, len(z)), replace=False).tolist())

        # 构建候选
        import itertools
        all_front_nums = list(set(hot_pool + cold_pool + balanced))
        if len(all_front_nums) < 7:
            all_front_nums = list(range(1, 36))

        for _ in range(n_candidates):
            selected_front = sorted(np.random.choice(all_front_nums, 5, replace=False).tolist())
            selected_back = sorted(np.random.choice(range(1, 13), 2, replace=False).tolist())
            candidates.append({
                'front': selected_front,
                'back': selected_back,
                'base_score': 0.5,
                'strategy_name': 'PoolSampler',
            })

        return candidates
