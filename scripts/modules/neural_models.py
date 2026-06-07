"""
神经网络模型模块 (neural_models.py)
TabNet / LSTM / Transformer 三模型集成 + 特征工程
"""

import os
import math
import pickle
import warnings
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    if hasattr(torch, '__version__') and int(torch.__version__.split('.')[0]) >= 1:
        TORCH_AVAILABLE = True
except ImportError:
    pass

# ============================================================
# 常量
# ============================================================
FRONT_N = 35
BACK_N = 12
FRONT_SELECT = 5
BACK_SELECT = 2
FRONT_PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
BACK_PRIMES = {2, 3, 5, 7, 11}
WINDOW = 50  # 特征构建窗口
SEQ_LEN = 20  # 序列模型输入步长
NUM_FEATURES = 12  # 每个号码的特征维度


# ============================================================
# 特征构建器
# ============================================================

class NeuralFeatureBuilder:
    """
    为神经网络构建特征张量。

    每种号码（前区1-35，后区1-12）构建多维特征向量，
    支持批量构建和序列构建两种模式。

    输入数据格式: draws = List[Tuple[List[int], List[int]]]
        每个元素 = (前区5个号码, 后区2个号码), 按时间升序排列
    """

    def __init__(self, draws: List[Tuple[List[int], List[int]]], window: int = WINDOW):
        self.draws = draws
        self.window = window
        self.front_mean = np.zeros(NUM_FEATURES)
        self.front_std = np.ones(NUM_FEATURES)
        self.back_mean = np.zeros(NUM_FEATURES)
        self.back_std = np.ones(NUM_FEATURES)
        # 拟合归一化参数
        self._fit_normalization()

    def _compute_single_features(self, draws_window: List[Tuple[List[int], List[int]]],
                                 num: int, zone: str = 'front') -> np.ndarray:
        """为单个号码构建12维特征向量"""
        n_max = FRONT_N if zone == 'front' else BACK_N
        primes = FRONT_PRIMES if zone == 'front' else BACK_PRIMES
        n_window = len(draws_window)
        if n_window == 0:
            return np.zeros(NUM_FEATURES, dtype=np.float32)

        # 1. 频率: 近window期出现次数/window
        freq = sum(1 for f, _ in draws_window if num in f) if zone == 'front' \
            else sum(1 for _, b in draws_window if num in b)
        freq_norm = freq / max(n_window, 1)

        # 2. 遗漏值: 距上次出现经过的期数 / window
        missing = n_window
        for i in range(n_window - 1, -1, -1):
            nums = draws_window[i][0] if zone == 'front' else draws_window[i][1]
            if num in nums:
                missing = n_window - 1 - i
                break
        missing_norm = min(missing / max(self.window, 1), 1.0)

        # 3. 趋势: 近5期 vs 近20期频率差值, 归一化到[0,1]
        recent5 = draws_window[-5:] if n_window >= 5 else draws_window
        recent20 = draws_window[-20:] if n_window >= 20 else draws_window
        f5 = sum(1 for f, _ in recent5 if num in f) / max(len(recent5), 1) if zone == 'front' \
            else sum(1 for _, b in recent5 if num in b) / max(len(recent5), 1)
        f20 = sum(1 for f, _ in recent20 if num in f) / max(len(recent20), 1) if zone == 'front' \
            else sum(1 for _, b in recent20 if num in b) / max(len(recent20), 1)
        trend = (f5 - f20 + 1.0) / 2.0

        # 4. 上期重号标记
        repeat = 0.0
        if n_window >= 1:
            last = draws_window[-1]
            last_nums = last[0] if zone == 'front' else last[1]
            repeat = 1.0 if num in last_nums else 0.0

        # 5. 隔期重号标记（上上期）
        skip = 0.0
        if n_window >= 2:
            skip_draw = draws_window[-2]
            skip_nums = skip_draw[0] if zone == 'front' else skip_draw[1]
            skip = 1.0 if num in skip_nums else 0.0

        # 6. 临号标记（与上期号码相邻）
        adjacent = 0.0
        if n_window >= 1:
            last = draws_window[-1]
            last_nums = set(last[0] if zone == 'front' else last[1])
            if (num - 1) in last_nums or (num + 1) in last_nums:
                adjacent = 1.0
            # 跨边界处理
            if num == 1 and 2 in last_nums:
                adjacent = 1.0
            if num == n_max and (n_max - 1) in last_nums:
                adjacent = 1.0

        # 7. 奇偶标记
        odd = 1.0 if num % 2 == 1 else 0.0

        # 8. 质数标记
        prime = 1.0 if num in primes else 0.0

        # 9. 区间位置归一化 (0~1)
        zone_pos = (num - 1) / (n_max - 1)

        # 10. 尾号 (个位数/9)
        tail = (num % 10) / 9.0

        # 11. 热号衰减: 加权频率，近期的权重更大
        weighted_freq = 0.0
        total_weight = 0.0
        for i in range(max(0, n_window - 30), n_window):
            weight = (i - max(0, n_window - 30) + 1) / 30.0
            nums = draws_window[i][0] if zone == 'front' else draws_window[i][1]
            if num in nums:
                weighted_freq += weight
            total_weight += weight
        weighted_norm = weighted_freq / max(total_weight, 1e-6)

        # 12. 间隔周期性: 上次出现至今的间隔是否接近历史模式
        # 计算所有出现间隔的CV(变异系数)
        appearances = []
        for i in range(n_window):
            nums = draws_window[i][0] if zone == 'front' else draws_window[i][1]
            if num in nums:
                appearances.append(i)
        if len(appearances) >= 3:
            gaps = np.diff(appearances)
            cv = np.std(gaps) / max(np.mean(gaps), 1e-6)
            regularity = 1.0 / (1.0 + min(cv, 10.0))  # 0~1, 越规律越接近1
        else:
            regularity = 0.3

        return np.array([
            freq_norm, missing_norm, trend, repeat, skip,
            adjacent, odd, prime, zone_pos, tail,
            weighted_norm, regularity
        ], dtype=np.float32)

    def _fit_normalization(self):
        """拟合归一化参数"""
        if len(self.draws) < self.window + 1:
            return
        front_feats = []
        back_feats = []
        for i in range(self.window, len(self.draws)):
            window_data = self.draws[i - self.window:i]
            for zone, feats_list in [('front', front_feats), ('back', back_feats)]:
                n_max = FRONT_N if zone == 'front' else BACK_N
                for num in range(1, n_max + 1):
                    feats_list.append(self._compute_single_features(window_data, num, zone))

        if front_feats:
            arr = np.array(front_feats)
            self.front_mean = arr.mean(axis=0)
            self.front_std = arr.std(axis=0).clip(min=1e-6)
        if back_feats:
            arr = np.array(back_feats)
            self.back_mean = arr.mean(axis=0)
            self.back_std = arr.std(axis=0).clip(min=1e-6)

    def build_candidate_features(self, front: List[int], back: List[int],
                                 draws: Optional[List[Tuple]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        为单个候选构建特征: 提取候选号码的特征向量

        Returns:
            front_feats: (5, NUM_FEATURES) ndarray
            back_feats: (2, NUM_FEATURES) ndarray
        """
        if draws is None:
            draws = self.draws
        window_data = draws[-min(self.window, len(draws)):] if len(draws) >= 2 else draws

        f_feats = np.array([self._compute_single_features(window_data, n, 'front') for n in front])
        b_feats = np.array([self._compute_single_features(window_data, n, 'back') for n in back])

        # 归一化
        f_feats = (f_feats - self.front_mean) / self.front_std
        b_feats = (b_feats - self.back_mean) / self.back_std

        return f_feats, b_feats

    def build_sequence(self, draws: Optional[List[Tuple]] = None,
                       seq_len: int = SEQ_LEN) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建序列数据用于LSTM/Transformer

        Returns:
            front_seq: (seq_len, FRONT_N, NUM_FEATURES) 每期的所有号码特征
            back_seq: (seq_len, BACK_N, NUM_FEATURES)
        """
        if draws is None:
            draws = self.draws
        if len(draws) < seq_len + 1:
            seq_len = max(1, len(draws) - 1)

        seq_data = draws[-seq_len - 1:-1]  # 最新的seq_len期（排除最后一期用作target）
        if len(seq_data) < seq_len:
            seq_data = draws[:seq_len]

        front_seq = np.zeros((seq_len, FRONT_N, NUM_FEATURES), dtype=np.float32)
        back_seq = np.zeros((seq_len, BACK_N, NUM_FEATURES), dtype=np.float32)

        for t in range(seq_len):
            window_data = draws[:max(0, len(draws) - seq_len + t)]
            if len(window_data) < 2:
                # 补充早期数据的窗口
                window_data = draws[:max(self.window, len(draws))]
            window_data = window_data[-min(self.window, len(window_data)):]

            for n in range(1, FRONT_N + 1):
                front_seq[t, n - 1] = self._compute_single_features(window_data, n, 'front')
            for n in range(1, BACK_N + 1):
                back_seq[t, n - 1] = self._compute_single_features(window_data, n, 'back')

        # 归一化
        front_seq = (front_seq - self.front_mean) / self.front_std
        back_seq = (back_seq - self.back_mean) / self.back_std

        return front_seq, back_seq

    def build_labels(self, draws: Optional[List[Tuple]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建标签: 最后一期的号码作为目标

        Returns:
            front_labels: (FRONT_N,) - 1表示该号码出现
            back_labels: (BACK_N,)
        """
        if draws is None:
            draws = self.draws
        if not draws:
            return np.zeros(FRONT_N), np.zeros(BACK_N)
        latest = draws[-1]
        f_labels = np.zeros(FRONT_N, dtype=np.float32)
        b_labels = np.zeros(BACK_N, dtype=np.float32)
        for n in latest[0]:
            f_labels[n - 1] = 1.0
        for n in latest[1]:
            b_labels[n - 1] = 1.0
        return f_labels, b_labels


# ============================================================
# PyTorch 模型定义
# ============================================================

if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """Transformer位置编码"""
        def __init__(self, d_model: int, max_len: int = 100):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2] if d_model % 2 == 0 else position * div_term[:d_model - d_model // 2])
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]


    class TabNetModel(nn.Module):
        """
        简化版TabNet: 注意力特征选择 + 全连接分类器。
        输入: (batch, NUM_FEATURES) 单号码特征
        输出: (batch,) sigmoid 概率
        """
        def __init__(self, input_dim: int = NUM_FEATURES, hidden_dim: int = 64):
            super().__init__()
            self.feature_transformer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            # 注意力特征选择
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.Softmax(dim=-1),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim + input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            # x: (batch, features)
            transformed = self.feature_transformer(x)  # (batch, hidden)
            attn_weights = self.attn(transformed)  # (batch, input_dim)
            attended = x * attn_weights  # 特征选择
            combined = torch.cat([transformed, attended], dim=-1)
            return torch.sigmoid(self.classifier(combined)).squeeze(-1)


    class LSTMModel(nn.Module):
        """
        双向LSTM + 注意力池化。
        输入: (batch, seq_len, n_numbers, features)
        输出: (batch, n_numbers) 概率
        """
        def __init__(self, input_dim: int = NUM_FEATURES, hidden_dim: int = 64,
                     num_layers: int = 2, n_numbers: int = FRONT_N):
            super().__init__()
            self.n_numbers = n_numbers
            self.hidden_dim = hidden_dim
            self.input_proj = nn.Linear(input_dim, hidden_dim)

            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.2 if num_layers > 1 else 0,
            )
            # 注意力
            self.attn_fc = nn.Linear(hidden_dim * 2, 1)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            # x: (batch, seq_len, n_numbers, features)
            batch, seq, nums, feats = x.shape
            # 合并 batch 和 nums 维度，对每个号码独立建模序列
            x = x.reshape(batch * nums, seq, feats)  # (batch*nums, seq, feats)
            x = self.input_proj(x)  # (batch*nums, seq, hidden)
            lstm_out, _ = self.lstm(x)  # (batch*nums, seq, hidden*2)

            # 注意力池化
            attn_scores = self.attn_fc(lstm_out).squeeze(-1)  # (batch*nums, seq)
            attn_weights = F.softmax(attn_scores, dim=-1)
            context = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch*nums, hidden*2)

            out = torch.sigmoid(self.classifier(context)).squeeze(-1)  # (batch*nums,)
            return out.reshape(batch, nums)


    class TransformerModel(nn.Module):
        """
        Transformer编码器 + 序列池化。
        输入: (batch, seq_len, n_numbers, features)
        输出: (batch, n_numbers) 概率
        """
        def __init__(self, input_dim: int = NUM_FEATURES, d_model: int = 64,
                     nhead: int = 4, num_layers: int = 2, n_numbers: int = FRONT_N):
            super().__init__()
            self.n_numbers = n_numbers
            self.d_model = d_model
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.2, activation='relu',
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x):
            # x: (batch, seq_len, n_numbers, features)
            batch, seq, nums, feats = x.shape
            x = x.reshape(batch * nums, seq, feats)
            x = self.input_proj(x) * math.sqrt(self.d_model)
            x = self.pos_encoder(x)
            x = self.transformer(x)  # (batch*nums, seq, d_model)

            # 取最后时间步的输出
            context = x[:, -1, :]  # (batch*nums, d_model)
            out = torch.sigmoid(self.classifier(context)).squeeze(-1)
            return out.reshape(batch, nums)


    # ============================================================
    # 训练器
    # ============================================================

    class _DLTDataset(Dataset):
        """从历史数据构建滑动窗口数据集"""
        def __init__(self, draws: List[Tuple], builder: NeuralFeatureBuilder,
                     zone: str = 'front', seq_len: int = SEQ_LEN):
            self.draws = draws
            self.builder = builder
            self.zone = zone
            self.seq_len = seq_len
            self.n_numbers = FRONT_N if zone == 'front' else BACK_N
            # 构建样本索引: 每个训练样本是 (seq_window, target_draw)
            self.sample_indices = []
            min_len = seq_len + 1
            for i in range(min_len, len(draws)):
                self.sample_indices.append(i)

        def __len__(self):
            return len(self.sample_indices)

        def __getitem__(self, idx):
            target_idx = self.sample_indices[idx]
            seq_start = target_idx - self.seq_len
            seq_data = self.draws[seq_start:target_idx]
            target = self.draws[target_idx]

            # 构建序列特征
            seq_feats = []
            for t in range(self.seq_len):
                window_data = self.draws[max(0, seq_start + t - self.builder.window):seq_start + t]
                if len(window_data) < 2:
                    window_data = self.draws[:max(2, self.builder.window)]
                feats_t = []
                n_max = self.n_numbers
                zone_name = 'front' if self.zone == 'front' else 'back'
                for n in range(1, n_max + 1):
                    feats_t.append(self.builder._compute_single_features(window_data, n, zone_name))
                seq_feats.append(feats_t)

            x = np.array(seq_feats, dtype=np.float32)  # (seq_len, n_numbers, features)
            mean = self.builder.front_mean if self.zone == 'front' else self.builder.back_mean
            std = self.builder.front_std if self.zone == 'front' else self.builder.back_std
            x = (x - mean) / std

            # 标签
            target_nums = target[0] if self.zone == 'front' else target[1]
            y = np.zeros(self.n_numbers, dtype=np.float32)
            for n in target_nums:
                y[n - 1] = 1.0

            return torch.FloatTensor(x), torch.FloatTensor(y)


    class BaseTrainer:
        """训练器基类"""
        def __init__(self, model: nn.Module, model_name: str, lr: float = 1e-3,
                     device: str = None):
            self.model = model
            self.model_name = model_name
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3
            )
            self.best_loss = float('inf')
            self.patience_counter = 0
            self.max_patience = 5

        def train_epoch(self, loader: DataLoader) -> float:
            self.model.train()
            total_loss = 0.0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x)
                # 多标签分类: BCE
                loss = F.binary_cross_entropy(pred, y, weight=torch.where(y > 0, 3.0, 1.0))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)
            return total_loss / len(loader.dataset)

        def validate(self, loader: DataLoader) -> float:
            self.model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = self.model(x)
                    loss = F.binary_cross_entropy(pred, y)
                    total_loss += loss.item() * x.size(0)
            return total_loss / len(loader.dataset)

        def fit(self, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 50, verbose: bool = True) -> Dict:
            history = {'train_loss': [], 'val_loss': []}
            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader)
                val_loss = self.validate(val_loader)
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"    [{self.model_name}] Epoch {epoch+1}/{epochs} "
                          f"train={train_loss:.4f} val={val_loss:.4f} "
                          f"(best={self.best_loss:.4f})")

                if self.patience_counter >= self.max_patience:
                    if verbose:
                        print(f"    [{self.model_name}] Early stopping at epoch {epoch+1}")
                    break

            return history

    class TabNetTrainer(BaseTrainer):
        """TabNet训练器"""
        def __init__(self, input_dim: int = NUM_FEATURES, hidden_dim: int = 64,
                     n_numbers: int = FRONT_N, lr: float = 1e-3):
            model = TabNetModel(input_dim, hidden_dim)
            super().__init__(model, 'TabNet', lr)
            self.input_dim = input_dim
            self.n_numbers = n_numbers

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            """
            输入: (n_numbers, features) 或 (batch, n_numbers, features)
            输出: (n_numbers,) 概率
            """
            self.model.eval()
            with torch.no_grad():
                if features.ndim == 2:
                    features = features[np.newaxis, :, :]  # (1, n, f) -> TabNet需要(batch, f) 独立处理，但我们先reshape
                batch, n, f = features.shape
                # TabNet处理每个号码独立，所以展平
                x = torch.FloatTensor(features.reshape(batch * n, f)).to(self.device)
                pred = self.model(x)  # (batch*n,)
                pred = pred.reshape(batch, n)
            return pred.cpu().numpy()[0] if batch == 1 else pred.cpu().numpy()


    class LSTMTrainer(BaseTrainer):
        """LSTM训练器"""
        def __init__(self, input_dim: int = NUM_FEATURES, hidden_dim: int = 64,
                     num_layers: int = 2, n_numbers: int = FRONT_N,
                     seq_len: int = SEQ_LEN, lr: float = 1e-3):
            model = LSTMModel(input_dim, hidden_dim, num_layers, n_numbers)
            super().__init__(model, 'LSTM', lr)
            self.input_dim = input_dim
            self.n_numbers = n_numbers
            self.seq_len = seq_len

        def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
            """
            输入: (seq_len, n_numbers, features)
            输出: (n_numbers,) 概率
            """
            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, seq, n, f)
                pred = self.model(x)  # (1, n)
            return pred.cpu().numpy()[0]


    class TransformerTrainer(BaseTrainer):
        """Transformer训练器"""
        def __init__(self, input_dim: int = NUM_FEATURES, d_model: int = 64,
                     nhead: int = 4, num_layers: int = 2, n_numbers: int = FRONT_N,
                     seq_len: int = SEQ_LEN, lr: float = 1e-3):
            model = TransformerModel(input_dim, d_model, nhead, num_layers, n_numbers)
            super().__init__(model, 'Transformer', lr)
            self.input_dim = input_dim
            self.n_numbers = n_numbers
            self.seq_len = seq_len

        def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
            """
            输入: (seq_len, n_numbers, features)
            输出: (n_numbers,) 概率
            """
            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                pred = self.model(x)
            return pred.cpu().numpy()[0]

else:
    # Torch不可用时的fallback
    _fallback_warned = False

    class _StubModel:
        def __init__(self, name: str):
            global _fallback_warned
            if not _fallback_warned:
                print(f"  ⚠️ [NeuralModels] PyTorch 不可用，{name} 使用 fallback")
                _fallback_warned = True
            self.name = name

        def predict_proba(self, *args, **kwargs) -> np.ndarray:
            return np.ones(FRONT_N) * 0.5

    class TabNetTrainer:
        def __init__(self, **kwargs):
            self._model = _StubModel('TabNet')
        def predict_proba(self, features):
            return self._model.predict_proba(features)

    class LSTMTrainer:
        def __init__(self, **kwargs):
            self._model = _StubModel('LSTM')
        def predict_proba(self, sequence):
            return self._model.predict_proba(sequence)

    class TransformerTrainer:
        def __init__(self, **kwargs):
            self._model = _StubModel('Transformer')
        def predict_proba(self, sequence):
            return self._model.predict_proba(sequence)


# ============================================================
# 神经网络集成器
# ============================================================

class NeuralEnsemble:
    """
    三模型神经网络集成器。

    使用策略:
    - TabNet: 基于单号码特征做分类（快速推理）
    - LSTM: 基于序列建模时序依赖
    - Transformer: 自注意力捕捉全局模式

    集成: 加权平均三个模型的输出
    """

    def __init__(self, draws: Optional[List[Tuple]] = None,
                 seq_len: int = SEQ_LEN, window: int = WINDOW,
                 train_epochs: int = 50, auto_train: bool = True):
        self.seq_len = seq_len
        self.window = window
        self.train_epochs = train_epochs
        self.is_trained = False
        self.train_history = {}
        self._ensemble_weights = {'tabnet': 0.25, 'lstm': 0.35, 'transformer': 0.40}

        # 特征构建器
        self.feature_builder = None
        # 模型
        self.tabnet_front = None
        self.tabnet_back = None
        self.lstm_front = None
        self.lstm_back = None
        self.transformer_front = None
        self.transformer_back = None

        if draws is not None and auto_train:
            self.train(draws)

    def _init_models(self, n_numbers: int, zone: str):
        """初始化指定区域的三个模型"""
        if zone == 'front':
            self.tabnet_front = TabNetTrainer(n_numbers=n_numbers)
            self.lstm_front = LSTMTrainer(n_numbers=n_numbers, seq_len=self.seq_len)
            self.transformer_front = TransformerTrainer(n_numbers=n_numbers, seq_len=self.seq_len)
        else:
            self.tabnet_back = TabNetTrainer(n_numbers=n_numbers)
            self.lstm_back = LSTMTrainer(n_numbers=n_numbers, seq_len=self.seq_len)
            self.transformer_back = TransformerTrainer(n_numbers=n_numbers, seq_len=self.seq_len)

    def train(self, draws: List[Tuple], verbose: bool = True) -> Dict:
        """在历史数据上训练所有模型"""
        if not TORCH_AVAILABLE:
            print("  ⚠️ [NeuralEnsemble] PyTorch不可用，跳过训练")
            self.is_trained = True
            return {}

        if len(draws) < self.seq_len + 10:
            print(f"  ⚠️ [NeuralEnsemble] 数据不足 ({len(draws)}期)，跳过训练")
            self.is_trained = True
            return {}

        print(f"  [NeuralEnsemble] 开始训练 ({len(draws)}期数据)...")

        # 构建特征器
        self.feature_builder = NeuralFeatureBuilder(draws, self.window)

        # 初始化模型
        self._init_models(FRONT_N, 'front')
        self._init_models(BACK_N, 'back')

        # 准备数据
        batch_size = min(64, max(8, len(draws) // 20))

        history = {}
        for zone, models in [('front', [
            (self.tabnet_front, 'TabNet-F'),
            (self.lstm_front, 'LSTM-F'),
            (self.transformer_front, 'TF-F'),
        ]), ('back', [
            (self.tabnet_back, 'TabNet-B'),
            (self.lstm_back, 'LSTM-B'),
            (self.transformer_back, 'TF-B'),
        ])]:
            if verbose:
                print(f"  ── {zone.upper()} 区 ──")

            for trainer, name in models:
                if verbose:
                    print(f"    训练 {name}...")
                dataset = _DLTDataset(draws, self.feature_builder, zone, self.seq_len)
                if len(dataset) < 10:
                    continue
                split = int(len(dataset) * 0.8)
                train_ds, val_ds = torch.utils.data.random_split(
                    dataset, [split, len(dataset) - split],
                    generator=torch.Generator().manual_seed(42),
                )
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size)
                hist = trainer.fit(train_loader, val_loader, epochs=self.train_epochs, verbose=verbose)
                history[f'{name}'] = hist

        self.is_trained = True
        self.train_history = history

        if verbose:
            print(f"  ✅ [NeuralEnsemble] 训练完成")

        return history

    def _compute_number_probs(self, draws: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算每个号码的出现概率。

        Returns:
            front_probs: (35,) ndarray - 前区每个号码的出现概率
            back_probs: (12,) ndarray - 后区每个号码的出现概率
        """
        if not self.is_trained:
            return np.ones(FRONT_N) * 0.5, np.ones(BACK_N) * 0.5

        if not TORCH_AVAILABLE:
            return np.ones(FRONT_N) * 0.5, np.ones(BACK_N) * 0.5

        fb = self.feature_builder

        # 构建序列数据
        front_seq, back_seq = fb.build_sequence(draws, self.seq_len)

        # TabNet: 使用最近窗口构建单号码特征
        window_data = draws[-min(self.window, len(draws)):]
        front_tabnet_feats = np.array([
            fb._compute_single_features(window_data, n, 'front')
            for n in range(1, FRONT_N + 1)
        ], dtype=np.float32)
        front_tabnet_feats = (front_tabnet_feats - fb.front_mean) / fb.front_std

        back_tabnet_feats = np.array([
            fb._compute_single_features(window_data, n, 'back')
            for n in range(1, BACK_N + 1)
        ], dtype=np.float32)
        back_tabnet_feats = (back_tabnet_feats - fb.back_mean) / fb.back_std

        # 推理
        front_probs = np.zeros(FRONT_N)
        back_probs = np.zeros(BACK_N)

        try:
            fp_t = self.tabnet_front.predict_proba(front_tabnet_feats[np.newaxis, :, :])
            fp_l = self.lstm_front.predict_proba(front_seq)
            fp_tf = self.transformer_front.predict_proba(front_seq)
            front_probs = (
                self._ensemble_weights['tabnet'] * fp_t +
                self._ensemble_weights['lstm'] * fp_l +
                self._ensemble_weights['transformer'] * fp_tf
            )
        except Exception as e:
            print(f"  ⚠️ [NeuralEnsemble] 前区推理失败: {e}")
            front_probs = np.ones(FRONT_N) * 0.5

        try:
            bp_t = self.tabnet_back.predict_proba(back_tabnet_feats[np.newaxis, :, :])
            bp_l = self.lstm_back.predict_proba(back_seq)
            bp_tf = self.transformer_back.predict_proba(back_seq)
            back_probs = (
                self._ensemble_weights['tabnet'] * bp_t +
                self._ensemble_weights['lstm'] * bp_l +
                self._ensemble_weights['transformer'] * bp_tf
            )
        except Exception as e:
            print(f"  ⚠️ [NeuralEnsemble] 后区推理失败: {e}")
            back_probs = np.ones(BACK_N) * 0.5

        return front_probs, back_probs

    def score_candidate(self, front: List[int], back: List[int],
                        draws: List[Tuple]) -> float:
        """
        对单个候选号码组合评分（0~1）。

        评分方法: 候选号码的平均对数概率
        - 取各号码概率的几何均值（对数空间平均）
        - 归一化到0~1
        """
        front_probs, back_probs = self._compute_number_probs(draws)

        # 候选号码的平均对数概率
        f_log_probs = [np.log(max(p, 1e-6)) for p in [front_probs[n - 1] for n in front]]
        b_log_probs = [np.log(max(p, 1e-6)) for p in [back_probs[n - 1] for n in back]]

        # 前区和后区加权组合
        f_score = np.exp(np.mean(f_log_probs))
        b_score = np.exp(np.mean(b_log_probs))
        combined = f_score * 0.7 + b_score * 0.3

        # 归一化: 随机选择的期望概率是 (5/35)^0.7 * (2/12)^0.3 ≈ 0.14
        # 用 sigmoid 风格映射到 0~1
        normalized = 1.0 / (1.0 + np.exp(-5.0 * (combined - 0.15)))

        return float(normalized)

    def score_batch(self, candidates: List[Dict], draws: List[Tuple]) -> List[Dict]:
        """
        批量评分候选组合。

        Args:
            candidates: [{'front': [5], 'back': [2]}, ...]
            draws: 历史数据

        Returns:
            同列表，每个元素增加'neural_score'字段
        """
        if not candidates:
            return candidates

        front_probs, back_probs = self._compute_number_probs(draws)

        for c in candidates:
            front = c.get('front', [])
            back = c.get('back', [])
            if len(front) != 5 or len(back) != 2:
                c['neural_score'] = 0.5
                continue

            f_log_probs = [np.log(max(front_probs[n - 1], 1e-6)) for n in front]
            b_log_probs = [np.log(max(back_probs[n - 1], 1e-6)) for n in back]

            f_score = np.exp(np.mean(f_log_probs))
            b_score = np.exp(np.mean(b_log_probs))
            combined = f_score * 0.7 + b_score * 0.3
            normalized = float(1.0 / (1.0 + np.exp(-5.0 * (combined - 0.15))))
            c['neural_score'] = normalized

        return candidates

    def save(self, path: str):
        """保存模型到文件"""
        state = {
            'ensemble_weights': self._ensemble_weights,
            'seq_len': self.seq_len,
            'window': self.window,
            'is_trained': self.is_trained,
            'train_history': self.train_history,
        }
        if TORCH_AVAILABLE and self.is_trained:
            for name in ['tabnet_front', 'tabnet_back', 'lstm_front', 'lstm_back',
                         'transformer_front', 'transformer_back']:
                model = getattr(self, name, None)
                if model and hasattr(model, 'model'):
                    state[f'{name}_state'] = model.model.state_dict()
            if self.feature_builder:
                state['front_mean'] = self.feature_builder.front_mean.tolist()
                state['front_std'] = self.feature_builder.front_std.tolist()
                state['back_mean'] = self.feature_builder.back_mean.tolist()
                state['back_std'] = self.feature_builder.back_std.tolist()

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str, draws: Optional[List[Tuple]] = None):
        """加载模型"""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            self._ensemble_weights = state.get('ensemble_weights', self._ensemble_weights)
            self.seq_len = state.get('seq_len', self.seq_len)
            self.window = state.get('window', self.window)
            self.is_trained = state.get('is_trained', False)
            self.train_history = state.get('train_history', {})

            if draws is not None:
                self.feature_builder = NeuralFeatureBuilder(draws, self.window)
                if 'front_mean' in state:
                    self.feature_builder.front_mean = np.array(state['front_mean'])
                    self.feature_builder.front_std = np.array(state['front_std'])
                    self.feature_builder.back_mean = np.array(state['back_mean'])
                    self.feature_builder.back_std = np.array(state['back_std'])

            if TORCH_AVAILABLE and self.is_trained:
                self._init_models(FRONT_N, 'front')
                self._init_models(BACK_N, 'back')
                for name in ['tabnet_front', 'tabnet_back', 'lstm_front', 'lstm_back',
                             'transformer_front', 'transformer_back']:
                    model = getattr(self, name, None)
                    if model and hasattr(model, 'model') and f'{name}_state' in state:
                        model.model.load_state_dict(state[f'{name}_state'])

            return True
        except Exception as e:
            print(f"  ⚠️ [NeuralEnsemble] 加载模型失败: {e}")
            return False


# Alias for framework compatibility
DLTFusionNet = NeuralEnsemble

__all__ = [
    'NeuralEnsemble',
    'NeuralFeatureBuilder',
    'TabNetTrainer',
    'LSTMTrainer',
    'TransformerTrainer',
    'FRONT_N',
    'BACK_N',
    'NUM_FEATURES',
    'SEQ_LEN',
]
