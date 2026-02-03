import timm
import gc
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import os
import re
import random
import argparse
import time
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------------
# 全局设置与工具函数
# -------------------------------------------------------------------------
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Try importing TPU libraries
try:
    import torch_xla

    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False


def seed_everything(seed: int = 42):
    """
    固定所有可能的随机种子，确保实验结果可复现。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """自动获取可用的计算设备 (GPU/CPU/TPU)。"""
    if TPU_AVAILABLE:
        return torch_xla.device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pct_to_class_6(pct: float) -> int:
    """
    将狭窄百分比转换为竞赛规定的 6 分类标签。
    映射规则: 0, 1-24, 25-49, 50-69, 70-99, 100
    """
    if pct <= 0:
        return 0
    if pct < 25:
        return 1
    if pct < 50:
        return 2
    if pct < 70:
        return 3
    if pct < 100:
        return 4
    return 5


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                # 确保只在 trace_func 允许的情况下打印 (即主进程)
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''验证集损失下降时保存模型'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # 使用全局必须存在的函数 (假设此时已定义)
        # 或者直接在这里实现类似的逻辑
        if TPU_AVAILABLE:
            if xr.global_ordinal() == 0:
                cpu_state_dict = {k: v.cpu()
                                  for k, v in model.state_dict().items()}
                torch.save(cpu_state_dict, self.path)
                del cpu_state_dict
        else:
            start_state_dict = model.module.state_dict() if isinstance(
                model, nn.DataParallel) else model.state_dict()
            torch.save(start_state_dict, self.path)

        self.val_loss_min = val_loss


def find_multiview_images(image_root: str, record_id: str, cache: Dict[str, List[str]] = {}) -> List[str]:
    """
    根据 record_id 查找对应的多视图图像文件。

    Args:
        image_root: 图像存储目录
        record_id: 样本ID
        cache: 目录文件列表缓存，避免重复IO操作

    Returns:
        按视图顺序排序的文件名列表
    """
    if image_root not in cache:
        if os.path.exists(image_root):
            with os.scandir(image_root) as it:
                cache[image_root] = [e.name for e in it if e.is_file()]
        else:
            cache[image_root] = []

    files = cache[image_root]
    rid = str(record_id).strip().lower()
    pattern = re.compile(r'view[_-]?(\d+)')

    candidates = []
    for fname in files:
        f_lower = fname.lower()
        # 简单匹配：文件名需包含ID且即为图片格式
        if rid in f_lower and f_lower.endswith(('.png', '.jpg', '.jpeg')):
            match = pattern.search(f_lower)
            if match:
                candidates.append((fname, int(match.group(1))))

    # 按视图编号 (view1, view2...) 升序排序
    return [c[0] for c in sorted(candidates, key=lambda x: x[1])]

# -------------------------------------------------------------------------
# 数据集定义
# -------------------------------------------------------------------------


class DabangDataset(Dataset):
    """
    多视图血管造影数据集。
    负责读取图像、预处理、对齐视图数量并返回 Tensor。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        num_views: int = 8,
        augment: bool = False,
        img_size: int = 224,
        is_test: bool = False
    ):
        self.image_root = image_root
        self.num_views = num_views
        self.is_test = is_test

        # 基础图像预处理: 调整大小 -> 转张量 -> 归一化
        # 假设输入为灰度图 (单通道)，使用 mean=0.5, std=0.5 进行各种归一化
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

        # 训练集数据增强: 温和增强策略 (水平翻转 + 微旋转 + 平移缩放)
        if augment:
            self.aug_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                # T.RandomRotation(degrees=15),  # 增加 +/- 15度旋转
                # T.RandomAffine(degrees=0, translate=(0.1, 0.1),
                #                scale=(0.9, 1.1)),  # 轻微平移和缩放
                # T.ColorJitter(brightness=0.1, contrast=0.1),  # 轻微亮度对比度变化
                # 暂时注释掉，后续根据需要调整
            ])
        else:
            self.aug_transform = None

        # 预先加载数据索引
        self.samples = self._load_index(df)

    def _load_index(self, df: pd.DataFrame) -> List[Tuple]:
        """构建样本索引列表，提前处理文件查找逻辑。"""
        # 只在主进程打印
        if (not TPU_AVAILABLE) or (TPU_AVAILABLE and xr.global_ordinal() == 0):
            print(f"正在构建数据集索引 ({len(df)} 样本)...")

        samples = []
        for _, row in df.iterrows():
            rid = row.get('record_id', row.get('uniq_ID'))
            if not rid:
                continue
            rid = str(rid).strip()

            # 1. 标签处理
            label = -1
            if not self.is_test:
                # 优先直接使用分类标签，缺失时通过百分比转换
                if pd.notna(row.get('stenosis_class')):
                    label = int(row['stenosis_class'])
                else:
                    pct = row.get('stenosis_percentage', 0)
                    label = pct_to_class_6(float(pct))

            # 2. 图像查找
            img_files = find_multiview_images(self.image_root, rid)
            if not img_files:
                continue

            # 3. 视图数量对齐 (截断或填充)
            curr = len(img_files)
            if curr > self.num_views:
                indices = np.linspace(0, curr - 1, self.num_views, dtype=int)
                img_files = [img_files[i] for i in indices]
            elif curr < self.num_views:
                img_files += [img_files[-1]] * (self.num_views - curr)

            paths = [os.path.join(self.image_root, f) for f in img_files]
            pid = str(row.get('pid', rid)).strip()
            samples.append((paths, label, rid, pid))

        if (not TPU_AVAILABLE) or (TPU_AVAILABLE and xr.global_ordinal() == 0):
            print(f"索引构建完成，有效样本数: {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label, rid, pid = self.samples[idx]
        tensors = []

        for p in paths:
            try:
                # 读取图像并转为单通道灰度图 ('L')
                img = Image.open(p).convert('L')

                # 应用增强
                if self.aug_transform:
                    img = self.aug_transform(img)

                t = self.transform(img)
                tensors.append(t)
            except Exception:
                # 异常容错: 返回全零张量
                tensors.append(torch.zeros((1, 224, 224)))

        # 核心逻辑: 将多视图在通道维度堆叠
        # 输入: List of [1, H, W] -> Output: [num_views, H, W]
        # 注意: 这里假设是单通道图片。如果是RGB，通道数需 x3
        x = torch.cat(tensors, dim=0)
        return x, label, rid, pid

# -------------------------------------------------------------------------
# 模型定义
# -------------------------------------------------------------------------


class BaselineModel(nn.Module):
    """
    比赛 Baseline 模型。
    直接使用 Timm 库创建模型，并修改输入通道数以支持多视图堆叠输入。
    不使用预训练权重，进行全量训练。
    """

    def __init__(self, model_name: str, num_classes: int, num_views: int):
        super().__init__()
        # 计算总输入通道数 (Views * 1 Channel)
        in_chans = num_views * 1

        # 使用 timm 直接创建模型
        # in_chans: 自动适配第一层卷积的输入通道
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_chans
        )

    def forward(self, x):
        return self.backbone(x)


class SiameseModel(nn.Module):
    """
    基于 Siamese 网络的特征融合模型。
    每个视角共用同一个 Backbone 提取特征，随后融合特征进行分类。
    使用 ImageNet 预训练权重。
    """

    def __init__(self, model_name: str, num_classes: int, num_views: int):
        super().__init__()
        self.num_views = num_views

        # 1. 创建共享的 Backbone (Siamese Network)
        # in_chans=1: 因为输入是单通道灰度图
        # num_classes=0: 只提取特征向量，不进行分类
        # pretrained=True: 使用 ImageNet 预训练权重加速收敛
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            in_chans=1
        )

        # 获取 Backbone 输出特征维度
        if hasattr(self.backbone, 'num_features'):
            feature_dim = self.backbone.num_features
        else:
            # 兼容部分没有 num_features 属性的模型
            dummy = torch.randn(1, 1, 224, 224)
            out = self.backbone(dummy)
            feature_dim = out.shape[1]

        # 2. 特征融合层与分类头
        # 融合策略: Flatten all views -> num_views * feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim * num_views, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: [Batch, Views, H, W] -> [B, 8, 224, 224]
        b, v, h, w = x.shape

        # 1. Reshape 为 [Batch * Views, 1, H, W] 以输入 Backbone
        # 添加 channel 维度 (unsqueeze(1) 没法直接用，因为 x 是 [B, V, H, W]，要变成 [B*V, 1, H, W])
        x = x.view(b * v, 1, h, w)

        # 2. 提取特征 -> [Batch * Views, Feature_Dim]
        features = self.backbone(x)

        # 3. 还原维度 -> [Batch, Views, Feature_Dim]
        features = features.view(b, v, -1)

        # 4. 特征融合 (Aggregation)
        # Flatten: [Batch, Views, Feature_Dim] -> [Batch, Views * Feature_Dim]
        combined = features.view(b, -1)

        # 5. 分类
        out = self.classifier(combined)
        return out


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2, bias=False),
            nn.Tanh(),
            nn.Linear(dim // 2, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [Batch, Views, Dim]
        # weights: [Batch, Views, 1]
        weights = self.attention(x)
        # weighted_feature: [Batch, Dim]
        return torch.sum(x * weights, dim=1)


class AttentionSiameseModel(nn.Module):
    """
    基于多视图特征融合的统一模型框架。
    支持 Attention / Mean / Max / Concat 等融合方式。
    """

    def __init__(self, model_name: str, num_classes: int, num_views: int, fusion_type: str = 'attention'):
        super().__init__()
        self.num_views = num_views
        self.fusion_type = fusion_type.lower()

        # Backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            in_chans=1
        )

        if hasattr(self.backbone, 'num_features'):
            feature_dim = self.backbone.num_features
        else:
            dummy = torch.randn(1, 1, 224, 224)
            out = self.backbone(dummy)
            feature_dim = out.shape[1]

        # Fusion Layer
        if self.fusion_type == 'attention':
            self.attn_pool = AttentionPool(feature_dim)
            classifier_input_dim = feature_dim
        elif self.fusion_type == 'concat':
            classifier_input_dim = feature_dim * num_views
        else:
            # Mean or Max
            classifier_input_dim = feature_dim

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3 if self.fusion_type != 'concat' else 0.5),
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        b, v, h, w = x.shape
        x = x.view(b * v, 1, h, w)
        features = self.backbone(x)
        features = features.view(b, v, -1)

        # Fusion Strategy
        if self.fusion_type == 'attention':
            # [Batch, Views, Dim] -> [Batch, Dim]
            pooled = self.attn_pool(features)
        elif self.fusion_type == 'mean':
            pooled = torch.mean(features, dim=1)
        elif self.fusion_type == 'max':
            pooled = torch.max(features, dim=1)[0]
        elif self.fusion_type == 'concat':
            pooled = features.view(b, -1)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        out = self.classifier(pooled)
        return out

# -------------------------------------------------------------------------
# 训练与验证流程
# -------------------------------------------------------------------------


def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def save_model_safely(model, path):
    """安全保存模型，尽量减少内存占用"""
    if TPU_AVAILABLE:
        # 只在主进程保存
        if xr.global_ordinal() == 0:
            try:
                # 显式将 state_dict 转移到 CPU，然后立刻删除 TPU 引用
                cpu_state_dict = {k: v.cpu()
                                  for k, v in model.state_dict().items()}
                torch.save(cpu_state_dict, path)
                del cpu_state_dict
                gc.collect()
            except Exception as e:
                print(f"Error saving model: {e}")
        # TPU 屏障
        # xm.rendezvous('save_model')
    else:
        torch.save(model.state_dict(), path)


def train_epoch(model, loader, optimizer, criterion, device, scaler=None, mixup_alpha=0.0):
    """单轮训练逻辑，支持 MixUp"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loader_wrapper = loader
    if TPU_AVAILABLE and device.type == 'xla':
        # 在TPU上，需要用ParallelLoader包装器
        loader_wrapper = pl.ParallelLoader(
            loader, [device]).per_device_loader(device)

    optimizer.zero_grad()

    # 仅在主进程或非TPU环境显示进度条
    show_pbar = (not TPU_AVAILABLE) or (
        TPU_AVAILABLE and xr.global_ordinal() == 0)

    if show_pbar:
        pbar = tqdm(loader_wrapper, desc="Train", leave=False)
        iter_loader = pbar
    else:
        iter_loader = loader_wrapper
        pbar = None

    for x, y, _, _ in iter_loader:
        x, y = x.to(device), y.to(device)

        # Mixup处理
        use_mixup = (mixup_alpha > 0)

        # TPU上暂时建议不使用 torch.cuda.amp.autocast, 或者使用特定的 amp
        if TPU_AVAILABLE:
            if use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha, device)
                outputs = model(x)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(x)
                loss = criterion(outputs, y)
        else:
            if use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha, device)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(x)
                        loss = mixup_criterion(
                            criterion, outputs, y_a, y_b, lam)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(x)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                    loss.backward()
                    optimizer.step()
            else:
                # 常规训练
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(x)
                        loss = criterion(outputs, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

        if TPU_AVAILABLE:
            loss.backward()
            optimizer.step()
            xm.mark_step()
            optimizer.zero_grad()
        else:
            # Non-TPU backward/step handled in the if/else blocks above for mixup/amp
            optimizer.zero_grad()

        # 统计指标
        with torch.no_grad():
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)

            if use_mixup:
                # Mixup 下的准确率近似计算
                # 避免 .cpu() 和 .item()以减少 TPU 同步
                if TPU_AVAILABLE:
                    correct += (lam * predicted.eq(y_a).float().sum() +
                                (1 - lam) * predicted.eq(y_b).float().sum())
                else:
                    correct += (lam * predicted.eq(y_a).cpu().sum().float() +
                                (1 - lam) * predicted.eq(y_b).cpu().sum().float()).item()
            else:
                if TPU_AVAILABLE:
                    correct += predicted.eq(y).float().sum()
                else:
                    correct += predicted.eq(y).sum().item()

        if pbar:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # TPU同步 Metrics
    if TPU_AVAILABLE:
        # Reduce metrics across cores for accurate logging
        # 注意: 如果 running_loss 是 tensor，需要确保它在 device 上
        if not isinstance(running_loss, torch.Tensor):
            running_loss = torch.tensor(running_loss, device=device)
        if not isinstance(correct, torch.Tensor):
            correct = torch.tensor(correct, device=device)

        t_metrics = torch.stack(
            [running_loss, correct, torch.tensor(total, device=device)])
        xm.all_reduce('sum', [t_metrics])
        running_loss = t_metrics[0].item()
        correct = t_metrics[1].item()
        total = t_metrics[2].item()

        # TPU world size correction for loader length if needed,
        # but here we just return mean loss
        loader_len = len(loader) * xr.world_size()
        return running_loss / (loader_len if loader_len > 0 else 1), correct / (total if total > 0 else 1)

    return running_loss / len(loader), correct / total


def validate(model, loader, device, criterion, num_classes=6):
    """验证集评估逻辑 (支持TPU多核Reduce，增加Kappa和F1)"""
    model.eval()
    running_loss = 0.0

    # 用于计算 Kappa 和 F1 的混淆矩阵 [num_classes, num_classes]
    # Rows: True, Cols: Pred
    confusion_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.long, device=device)

    total = 0

    loader_wrapper = loader
    if TPU_AVAILABLE and device.type == 'xla':
        loader_wrapper = pl.ParallelLoader(
            loader, [device]).per_device_loader(device)

    show_pbar = (not TPU_AVAILABLE) or (
        TPU_AVAILABLE and xr.global_ordinal() == 0)
    iter_loader = tqdm(loader_wrapper, desc="Val",
                       leave=False) if show_pbar else loader_wrapper

    with torch.no_grad():
        for x, y, _, _ in iter_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            # 累计 Loss
            running_loss += loss.item()

            # 更新混淆矩阵
            _, preds = outputs.max(1)
            # y和preds都是 [Batch]
            # 快速计算混淆矩阵的技巧: y * num_classes + preds
            # 这种方式对于分布式 reduce 非常友好，因为 matrix 很小
            indices = y * num_classes + preds
            batch_conf_mat = torch.bincount(indices, minlength=num_classes**2)
            confusion_matrix += batch_conf_mat.view(num_classes, num_classes)

            total += y.size(0)

    # TPU 同步聚合
    if TPU_AVAILABLE:
        if not isinstance(running_loss, torch.Tensor):
            running_loss = torch.tensor(running_loss, device=device)

        # 聚合 Loss 和 混淆矩阵
        # xm.all_reduce 支持 list，一次性 reduce 效率更高
        t_metrics = [running_loss, confusion_matrix]
        xm.all_reduce('sum', t_metrics)

        global_loss_sum = t_metrics[0].item()
        global_conf_mat = t_metrics[1].cpu().numpy()  # [6, 6]

        # 计算全局平均 Loss
        total_batches = len(loader) * xr.world_size()
        avg_loss = global_loss_sum / \
            (total_batches if total_batches > 0 else 1)

    else:
        avg_loss = running_loss / len(loader)
        global_conf_mat = confusion_matrix.cpu().numpy()



    # Accuracy: 对角线之和 / 总数
    tp_sum = np.trace(global_conf_mat)
    total_samples = np.sum(global_conf_mat)
    avg_acc = tp_sum / total_samples if total_samples > 0 else 0

    y_true_r = np.repeat(np.arange(num_classes), np.sum(
        global_conf_mat, axis=1).astype(int))
    # y_pred 从混淆矩阵恢复比较麻烦，每一行(True Class i)里，预测为各位 j 的数量
    y_pred_r_list = []
    for i in range(num_classes):
        # 对于真实类别 i，预测分布为 row i
        row = global_conf_mat[i]
        for j, count in enumerate(row):
            if count > 0:
                y_pred_r_list.extend([j] * int(count))
    # 注意: y_true_r 的生成顺序是 0,0,0...1,1,1...
    # y_pred_r_list生成顺序也是先遍历 True Class 0 里面的各个 Pred j，所以顺序是对齐的

    y_pred_r = np.array(y_pred_r_list)

    # 计算 Kappa (Quadratic)
    kappa = cohen_kappa_score(y_true_r, y_pred_r, weights='quadratic')

    # 计算 F1 Macro
    f1 = f1_score(y_true_r, y_pred_r, average='macro')

    return {
        'loss': avg_loss,
        'acc': avg_acc,
        'f1': f1,
        'kappa': kappa
    }

# -------------------------------------------------------------------------
# 参数配置与主程序
# -------------------------------------------------------------------------


dir_dataset = "/kaggle/input/cad-rads"


def print_config(args):
    """打印运行配置参数"""
    print("\n" + "="*40)
    print(f"{'Running Configuration':^40}")
    print("="*40)
    for key, value in sorted(vars(args).items()):
        print(f"{key:<20} : {value}")
    print("="*40 + "\n")


def run_inference(model_path, output_name, test_loader, model, device, args):
    if not os.path.exists(model_path):
        if (not TPU_AVAILABLE) or (xr.global_ordinal() == 0):
            print(
                f"Warning: Model path {model_path} does not exist. Skipping.")
        return

    if (not TPU_AVAILABLE) or (xr.global_ordinal() == 0):
        print(f"\n正在加载模型进行预测: {model_path}")

    # TPU上load需要注意，最好在cpu load然后to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()

    results = []

    loader_wrapper = test_loader
    if TPU_AVAILABLE and device.type == 'xla':
        loader_wrapper = pl.ParallelLoader(
            test_loader, [device]).per_device_loader(device)

    show_pbar = (not TPU_AVAILABLE) or (
        TPU_AVAILABLE and xr.global_ordinal() == 0)
    iter_loader = tqdm(
        loader_wrapper, desc=f"Inference ({output_name})") if show_pbar else loader_wrapper

    with torch.no_grad():
        for x, _, rids, pids in iter_loader:
            x = x.to(device)
            outputs = model(x)

            # TTA: 水平翻转预测 (最后两个维度是 H, W)
            x_flip = torch.flip(x, dims=[-1])
            outputs_flip = model(x_flip)
            outputs = (outputs + outputs_flip) / 2

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()

            for rid, pid, pred, prob in zip(rids, pids, preds, probs):
                row = {'ID': rid, 'Prediction': pred}
                # 保存每个类别的概率
                for i, p in enumerate(prob):
                    row[f'prob_{i}'] = p
                results.append(row)

    if TPU_AVAILABLE:
        # Save per-process results
        rank = xr.global_ordinal()
        part_csv = os.path.join(args.output_dir, f"part_{rank}_{output_name}")
        pd.DataFrame(results).to_csv(part_csv, index=False)

        # Sync
        xm.rendezvous('inference_save_done')

        if rank == 0:
            print(
                f"Merging inference results from all ranks for {output_name}...")
            all_parts = []
            world_size = xr.world_size()
            for r in range(world_size):
                fname = os.path.join(
                    args.output_dir, f"part_{r}_{output_name}")
                if os.path.exists(fname):
                    all_parts.append(pd.read_csv(fname))

            if all_parts:
                full_df = pd.concat(all_parts, ignore_index=True)
                out_csv = os.path.join(args.output_dir, output_name)
                full_df.to_csv(out_csv, index=False)
                print(f"完整预测结果已保存至: {out_csv}")

                # Cleanup
                for r in range(world_size):
                    fname = os.path.join(
                        args.output_dir, f"part_{r}_{output_name}")
                    if os.path.exists(fname):
                        os.remove(fname)

        # Sync again to prevent others from proceeding before merge is done
        xm.rendezvous('inference_merge_done')
    else:
        out_csv = os.path.join(args.output_dir, output_name)
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"预测结果已保存至: {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Dabang Competition Baseline")
    # 数据路径配置
    parser.add_argument(
        '--train_csv', default=f"{dir_dataset}/datasets/train_val.csv", help="训练集标签文件")
    parser.add_argument(
        '--test_csv', default=f"{dir_dataset}/datasets/test_competition.csv", help="测试集提交文件")
    parser.add_argument(
        '--img_root', default=f"{dir_dataset}/datasets/train_val", help="图像数据根目录")
    parser.add_argument('--test_img_root',
                        default=f"{dir_dataset}/datasets/test", help="测试图像目录(可选)")
    parser.add_argument('--output_dir', default="./results", help="结果输出目录")

    # 模型参数
    parser.add_argument('--model', default='convnext_tiny',
                        help='模型架构名称 (支持 timm)')
    parser.add_argument('--num_classes', type=int, default=6, help='分类类别数')
    parser.add_argument('--num_views', type=int, default=8, help='每个样本的视图数量')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early Stopping patience')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42, help='全局随机种子')
    parser.add_argument('--mixup_alpha', type=float,
                        default=0, help='MixUp alpha 参数 (0表示禁用)')
    parser.add_argument('--fusion', type=str, default='concat',
                        choices=['attention', 'mean', 'max', 'concat'],
                        help='多视图融合策略: attention, mean, max, concat (legacy siamese)')


    # Jupyter kernel 兼容参数
    parser.add_argument('-f', '--file', type=str,
                        required=False, help='Jupyter kernel file (ignore)')

    args, _ = parser.parse_known_args()
    return args


def run_worker(rank, args):
    """统一的训练工作流，支持单机和TPU多进程"""
    is_tpu = (rank is not None) and TPU_AVAILABLE

    # 1. 环境初始化
    seed_everything(args.seed)

    if is_tpu:
        device = torch_xla.device()
        is_master = (xr.global_ordinal() == 0)
    else:
        device = get_device()
        is_master = True

    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)
        print_config(args)
        prefix = "TPU Core" if is_tpu else "Device"
        dev_id = rank if is_tpu else device
        print(f"当前使用的计算设备: {prefix} {dev_id}")

    # 2. 数据准备
    if is_master:
        print("正在加载数据列表...")

    train_val_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv) if os.path.exists(
        args.test_csv) else pd.DataFrame()

    # PID 提取: 用于按病人划分验证集，防止数据泄漏
    for df in [train_val_df, test_df]:
        if df.empty:
            continue
        if 'pid' not in df.columns:
            col = 'check_id' if 'check_id' in df.columns else 'record_id'
            # 提取字符串中的数字部分作为 PID
            df['pid'] = df[col].astype(str).apply(lambda x: re.findall(
                r'\d+', x)[0] if re.findall(r'\d+', x) else x)

    # 划分训练集与验证集 (Ratio 85:15)
    pids = train_val_df['pid'].unique()
    train_pids, val_pids = train_test_split(
        pids, test_size=0.15, random_state=args.seed)
    train_df = train_val_df[train_val_df['pid'].isin(train_pids)].copy()
    val_df = train_val_df[train_val_df['pid'].isin(val_pids)].copy()

    # 处理测试集图像路径 (如果测试集在不同文件夹)
    test_root = args.test_img_root if os.path.exists(
        args.test_img_root) else args.img_root

    # 构建 Dataset
    # 注意: DabangDataset 内部会根据 global_ordinal 打印日志
    train_ds = DabangDataset(train_df, args.img_root,
                             num_views=args.num_views, augment=True)
    val_ds = DabangDataset(val_df, args.img_root,
                           num_views=args.num_views, augment=False)

    if not test_df.empty:
        test_ds = DabangDataset(
            test_df, test_root, num_views=args.num_views, is_test=True)
    else:
        test_ds = None

    # 构建 DataLoader
    # TPU 环境下不需要 pin_memory，甚至可能引发警告
    loader_args = {'batch_size': args.batch_size,
                   'num_workers': args.num_workers, 'pin_memory': not is_tpu}

    if is_tpu:
        # TPU Samplers
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=xr.world_size(), rank=xr.global_ordinal(), shuffle=True, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_ds, num_replicas=xr.world_size(), rank=xr.global_ordinal(), shuffle=False, drop_last=False)

        train_loader = DataLoader(
            train_ds, sampler=train_sampler, **loader_args)
        val_loader = DataLoader(val_ds, sampler=val_sampler, **loader_args)

        if test_ds:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_ds, num_replicas=xr.world_size(), rank=xr.global_ordinal(), shuffle=False, drop_last=False)
            test_loader = DataLoader(
                test_ds, sampler=test_sampler, **loader_args)
        else:
            test_loader = []

    else:
        # Standard Loaders
        train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
        test_loader = DataLoader(
            test_ds, shuffle=False, **loader_args) if test_ds else []

    # 3. 模型与优化器初始化
    # 使用统一的 AttentionSiameseModel，通过 fusion_type 参数控制
    fusion_strategy = args.fusion

    if is_master:
        print(f"初始化模型... Backend: {args.model}, Fusion: {fusion_strategy}")

    model = AttentionSiameseModel(
        model_name=args.model,
        num_classes=args.num_classes,
        num_views=args.num_views,
        fusion_type=fusion_strategy
    )

    if (not is_tpu) and (torch.cuda.device_count() > 1):
        if is_master:
            print(f"检测到 {torch.cuda.device_count()} 个 GPU，启用 DataParallel 并行训练")
        model = nn.DataParallel(model)

    model.to(device)

    # 优化器与损失函数
    # 引入 label_smoothing=0.1 缓解过拟合，对于分类边界模糊的医学图像通常有效
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Cosine Annealing Scheduler (T_max=epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # 混合精度 Scaler (非 TPU 环境)
    scaler = torch.cuda.amp.GradScaler() if (
        not is_tpu and torch.cuda.is_available()) else None

    # Helper for Master Printing
    def master_print(*args_p, **kwargs_p):
        if is_master:
            print(*args_p, **kwargs_p)

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=os.path.join(args.output_dir, "best_model_loss.pt"),
        trace_func=master_print
    )

    # TensorBoard & Logging
    writer = None
    if is_master:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(
                log_dir=os.path.join(args.output_dir, "logs"))
        except ImportError:
            print("Warning: tensorboard not installed.")

    log_history = []
    best_acc = 0.0
    best_kappa = -1.0
    log_csv_path = os.path.join(args.output_dir, "training_log.csv")

    if is_master:
        print("开始训练流程...")

    for epoch in range(args.epochs):
        t0 = time.time()

        if is_tpu:
            train_sampler.set_epoch(epoch)

        # 训练阶段
        # 注意：num_classes 默认是 args.num_classes，这里假设 validate 函数内部能获取到正确的类别数
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, mixup_alpha=args.mixup_alpha)

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 验证阶段
        val_metrics = validate(model, val_loader, device,
                               criterion, num_classes=args.num_classes)

        dt = time.time() - t0

        if is_master:
            print(f"Epoch {epoch+1:02d} ({dt:.1f}s) lr={current_lr:.2e} | "
                  f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                  f"Val: Loss={val_metrics['loss']:.4f} Acc={val_metrics['acc']:.4f} "
                  f"F1={val_metrics['f1']:.4f} Kappa={val_metrics['kappa']:.4f}")

            # 记录 CSV 日志
            log_history.append({
                'epoch': epoch + 1,
                'lr': current_lr,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['acc'],
                'val_f1': val_metrics['f1'],
                'val_kappa': val_metrics['kappa']
            })
            pd.DataFrame(log_history).to_csv(log_csv_path, index=False)

            # 记录 TensorBoard
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                writer.add_scalar('Accuracy/val', val_metrics['acc'], epoch)
                writer.add_scalar('F1/val', val_metrics['f1'], epoch)
                writer.add_scalar('Kappa/val', val_metrics['kappa'], epoch)
                writer.add_scalar('LR', current_lr, epoch)
                writer.flush()

            # 保存最佳模型 (以 Accuracy 为准)
            if val_metrics['acc'] > best_acc:
                best_acc = val_metrics['acc']
                save_path = os.path.join(args.output_dir, "best_model.pt")
                if is_tpu:
                    cpu_state_dict = {k: v.cpu()
                                      for k, v in model.state_dict().items()}
                    torch.save(cpu_state_dict, save_path)
                    del cpu_state_dict
                else:
                    state_dict = model.module.state_dict() if isinstance(
                        model, nn.DataParallel) else model.state_dict()
                    torch.save(state_dict, save_path)
                print(f"  >>> [Acc] 性能提升，模型已保存至: {save_path}")

            # 保存最佳模型 (以 Kappa 为准)
            if val_metrics['kappa'] > best_kappa:
                best_kappa = val_metrics['kappa']
                save_path_kappa = os.path.join(
                    args.output_dir, "best_model_kappa.pt")
                if is_tpu:
                    cpu_state_dict = {k: v.cpu()
                                      for k, v in model.state_dict().items()}
                    torch.save(cpu_state_dict, save_path_kappa)
                    del cpu_state_dict
                else:
                    state_dict = model.module.state_dict() if isinstance(
                        model, nn.DataParallel) else model.state_dict()
                    torch.save(state_dict, save_path_kappa)
                print(
                    f"  >>> [Kappa] 性能提升 ({best_kappa:.4f})，模型已保存至: {save_path_kappa}")

        # Early Stopping Check (Based on Val Loss)
        early_stopping(val_metrics['loss'], model)

        stop_flag = False
        if early_stopping.early_stop:
            stop_flag = True

        if is_tpu:
            # Sync stop flag
            flag_tensor = torch.tensor(
                1 if stop_flag else 0, dtype=torch.int, device=device)
            xm.all_reduce('max', [flag_tensor])
            stop_flag = (flag_tensor.item() > 0)

        if stop_flag:
            if is_master:
                print("Early stopping triggered!")
            break

    if is_master and writer:
        writer.close()

    if test_loader:
        if is_master:
            print("\n开始测试集预测...")

        # 1. 预测 Best Accuracy 模型
        best_acc_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(best_acc_path):
            run_inference(best_acc_path, "test_predictions_acc.csv",
                          test_loader, model, device, args)

        # 2. 预测 Best Kappa 模型
        best_kappa_path = os.path.join(args.output_dir, "best_model_kappa.pt")
        if os.path.exists(best_kappa_path):
            run_inference(best_kappa_path, "test_predictions_kappa.csv",
                          test_loader, model, device, args)

        # 3. 预测 Best Loss 模型
        best_loss_path = os.path.join(args.output_dir, "best_model_loss.pt")
        if os.path.exists(best_loss_path):
            run_inference(best_loss_path, "test_predictions_loss.csv",
                          test_loader, model, device, args)


def main():
    args = parse_args()

    if TPU_AVAILABLE:
        print("检测到 TPU 环境，启动多进程训练...")
        # 环境变量清理
        os.environ.pop('TPU_PROCESS_ADDRESSES', None)
        os.environ.pop('TPU_MESH_CONTROLLER_ADDRESS', None)
        os.environ.pop('TPU_MESH_CONTROLLER_PORT', None)

        xmp.spawn(run_worker, args=(args,), start_method='fork')
    else:
        run_worker(None, args)


if __name__ == "__main__":
    main()
