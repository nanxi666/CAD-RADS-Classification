import os
import re
import random
import argparse
import time
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import timm
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
# 忽略所有警告
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# 全局设置与工具函数
# -------------------------------------------------------------------------
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def _configure_warnings():
    """仅过滤已知可忽略的警告，保留其他重要提示。"""
    warnings.filterwarnings(
        "ignore",
        message=r".*invalid escape sequence.*",
        category=SyntaxWarning,
        module=r"torch_xla.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*tensorflow.*torch-xla.*",
        category=UserWarning,
        module=r"torch_xla.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Transparent hugepages are not enabled.*",
        category=UserWarning,
        module=r"jax.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Use torch_xla\.sync instead xm\.mark_step.*",
        category=DeprecationWarning,
    )


def _is_tpu_env() -> bool:
    """判断是否处在 TPU 运行环境。"""
    tpu_env_keys = (
        "COLAB_TPU_ADDR",
        "TPU_NAME",
        "TPU_IP_ADDRESS",
        "XRT_TPU_CONFIG",
        "TPU_WORKER_ID",
        "TPU_ACCELERATOR_TYPE",
        "TPU_VISIBLE_DEVICES",
        "PJRT_DEVICE",
    )
    for k in tpu_env_keys:
        v = os.environ.get(k, "")
        if v and (k != "PJRT_DEVICE" or v.upper() == "TPU"):
            return True
    return False


def _try_enable_transparent_hugepages():
    """尽力启用 Transparent Hugepages；若无权限则静默跳过。"""
    if os.name != "posix":
        return
    thp_path = "/sys/kernel/mm/transparent_hugepage/enabled"
    if not os.path.exists(thp_path):
        return
    try:
        with open(thp_path, "r", encoding="utf-8") as f:
            status = f.read()
        if "[always]" in status:
            return
        if os.access(thp_path, os.W_OK):
            with open(thp_path, "w", encoding="utf-8") as f:
                f.write("always")
    except Exception:
        return


_configure_warnings()

if _is_tpu_env():
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.runtime as xr
        TPU_AVAILABLE = True
        _try_enable_transparent_hugepages()
    except ImportError:
        TPU_AVAILABLE = False
else:
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


def try_enable_sync_batchnorm(model: nn.Module, is_tpu: bool, is_master: bool = True) -> nn.Module:
    """在 TPU 上尝试启用 SyncBatchNorm，失败则回退原模型。"""
    if not is_tpu:
        return model
    try:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_master:
            print("已启用 SyncBatchNorm")
        return model
    except Exception as exc:
        if is_master:
            print(f"SyncBatchNorm 启用失败，保持原 BN: {exc}")
        return model


def freeze_batchnorm(model: nn.Module, is_master: bool = True) -> None:
    """冻结所有 BN 统计与参数，避免小 batch 下不稳定。"""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters(recurse=False):
                p.requires_grad = False
    if is_master:
        print("已冻结 BatchNorm")


def replace_batchnorm_with_groupnorm(model: nn.Module, num_groups: int = 16, is_master: bool = True) -> nn.Module:
    """用 GroupNorm 替换 BatchNorm，适配小 batch 训练。"""
    def _convert(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                num_channels = child.num_features
                groups = min(num_groups, num_channels)
                while groups > 1 and (num_channels % groups != 0):
                    groups -= 1
                gn = nn.GroupNorm(groups, num_channels,
                                  eps=child.eps, affine=True)
                setattr(module, name, gn)
            else:
                _convert(child)

    _convert(model)
    if is_master:
        print("已将 BatchNorm 替换为 GroupNorm")
    return model


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


def compute_class_weights_from_df(df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """根据训练集统计类别权重（逆频率），用于处理类别不平衡。"""
    labels = []
    if 'multi_Class' in df.columns and df['multi_Class'].notna().any():
        labels = df['multi_Class'].dropna().astype(int).tolist()
    else:
        if 'stenosis_percentage' in df.columns:
            labels = [pct_to_class_6(
                float(p)) for p in df['stenosis_percentage'].fillna(0).tolist()]
    if len(labels) == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    counts = np.bincount(np.array(labels, dtype=int),
                         minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_warmup_cosine_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    """线性 warmup + 余弦退火调度器（epoch 级）。"""
    warmup_epochs = max(0, min(warmup_epochs, total_epochs))
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=1e-6,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


class EarlyStopping:
    """早停机制（基于验证集宏F1，越大越好）"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = -np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_f1, model):
        score = val_f1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
            self.counter = 0

    def save_checkpoint(self, val_f1, model):
        '''验证集宏F1提升时保存模型'''
        if self.verbose:
            self.trace_func(
                f'Validation F1 increased ({self.best_metric:.6f} --> {val_f1:.6f}).  Saving model ...')
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
        self.best_metric = val_f1


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
        self.vessel_map = {
            'LAD': 1,
            'LCX': 2,
            'RCA': 3,
        }

        # 基础数据增强与预处理 (Albumentations)
        # 注意: Albumentations 处理的是 numpy array (H, W, C)

        # 训练集增强策略
        if augment:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                # 几何变换
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625,
                                   scale_limit=0.1, rotate_limit=15, p=0.5),

                # 像素级变换 (适度使用，避免破坏血管细节)
                A.RandomBrightnessContrast(p=0.2),
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=0.2),
                    A.MotionBlur(blur_limit=3, p=0.2),
                ], p=0.3),

                # 模拟遮挡/噪声
                A.CoarseDropout(max_holes=8, max_height=img_size //
                                10, max_width=img_size//10, p=0.2),

                # 归一化 (使用 ImageNet 统计数据)
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            # 验证/测试集: 仅调整大小和归一化
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

        # 预先加载数据索引
        self.samples = self._load_index(df)

    def _load_index(self, df: pd.DataFrame) -> List[Tuple]:
        """构建样本索引列表，提前处理文件查找逻辑。"""
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
            vessel_code = str(row.get('vessel_code', '')).upper().strip()
            vessel_idx = self.vessel_map.get(vessel_code, 0)
            samples.append((paths, label, vessel_idx, rid, pid))

        if (not TPU_AVAILABLE) or (TPU_AVAILABLE and xr.global_ordinal() == 0):
            print(f"索引构建完成，有效样本数: {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label, vessel_idx, rid, pid = self.samples[idx]
        tensors = []

        for p in paths:
            try:
                # 读取图像并转为 Numpy 数组，强制转换为 RGB
                img_pil = Image.open(p).convert('RGB')
                img_np = np.array(img_pil)  # shape (H, W, 3)

                # 应用 Albumentations
                augmented = self.transform(image=img_np)
                t = augmented['image']  # Tensor [3, H, W]

                tensors.append(t)
            except Exception:
                # 异常容错: 返回全零张量
                tensors.append(torch.zeros((3, 224, 224)))

        # 堆叠视图 -> [Num_Views, 3, H, W]
        x = torch.stack(tensors, dim=0)
        return x, label, vessel_idx, rid, pid

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
    视图融合：Transformer Encoder (视图级序列建模)。
    """

    def __init__(self, model_name: str, num_classes: int, num_views: int,
                 use_cls_token: bool = False, use_vessel_code: bool = False, vessel_emb_dim: int = 16):
        super().__init__()
        self.num_views = num_views
        self.use_cls_token = use_cls_token
        self.use_vessel_code = use_vessel_code
        self.vessel_emb_dim = vessel_emb_dim

        # 1. 创建共享的 Backbone (Siamese Network)
        # in_chans=3: RGB 输入
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            in_chans=3,
            drop_path_rate=0
        )

        # 获取 Backbone 输出特征维度
        if hasattr(self.backbone, 'num_features'):
            feature_dim = self.backbone.num_features
        else:
            # 兼容部分没有 num_features 属性的模型
            dummy = torch.randn(1, 1, 224, 224)
            out = self.backbone(dummy)
            feature_dim = out.shape[1]

        # 2. 视图级 Transformer 融合
        token_len = num_views + (1 if self.use_cls_token else 0)
        self.view_pos = nn.Parameter(torch.zeros(1, token_len, feature_dim))
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

        def _pick_nhead(dim: int, max_head: int = 8) -> int:
            head = min(max_head, dim)
            while head > 1 and dim % head != 0:
                head -= 1
            return head

        nhead = _pick_nhead(feature_dim, max_head=8)
        self.view_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=nhead,
                dim_feedforward=feature_dim * 4,
                dropout=0.3,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=2,
        )

        # 3. vessel_code 条件向量
        if self.use_vessel_code:
            # 0=unknown, 1=LAD, 2=LCX, 3=RCA
            self.vessel_emb = nn.Embedding(4, vessel_emb_dim)

        # 4. 分类头 (输入为融合后的 feature_dim [+ vessel_emb_dim])
        in_dim = feature_dim + (vessel_emb_dim if self.use_vessel_code else 0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, vessel_idx=None):
        # x shape: [Batch, Views, 3, H, W]
        b, v, c, h, w = x.shape

        # 1. Reshape 为 [Batch * Views, C, H, W] 以输入 Backbone
        x = x.view(b * v, c, h, w)

        # 2. 提取特征 -> [Batch * Views, Feature_Dim]
        features = self.backbone(x)

        # 3. 还原维度 -> [Batch, Views, Feature_Dim]
        features = features.view(b, v, -1)

        # 4. Transformer 视图融合
        if self.use_cls_token:
            cls_tok = self.cls_token.expand(b, -1, -1)
            features = torch.cat([cls_tok, features], dim=1)

        features = features + self.view_pos
        fused = self.view_encoder(features)
        if self.use_cls_token:
            pooled = fused[:, 0]
        else:
            pooled = fused.mean(dim=1)

        # 5. 拼接 vessel_code 条件向量
        if self.use_vessel_code:
            if vessel_idx is None:
                vessel_feat = torch.zeros(
                    (b, self.vessel_emb_dim), device=pooled.device)
            else:
                vessel_idx = vessel_idx.to(
                    pooled.device).long().clamp(min=0, max=3)
                vessel_feat = self.vessel_emb(vessel_idx)
            pooled = torch.cat([pooled, vessel_feat], dim=1)

        # 6. 分类
        out = self.classifier(pooled)
        return out

# -------------------------------------------------------------------------
# 训练与验证流程
# -------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """
    Focal Loss = -alpha * (1-pt)^gamma * log(pt)
    修正版: 正确处理 class_weights 和 label_smoothing
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # inputs: [N, C] (logits)
        # targets: [N] (indices)

        # 1. 计算 Cross Entropy Loss (不带任何权重，仅单纯计算 Loss 用于梯度回传)
        # 这里为了配合 Focal Term，先不加 weight，手动应用
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', label_smoothing=self.label_smoothing
        )

        # 2. 计算 pt (真实类别的预测概率) 用于 Focal Modulation
        # pt 必须基于纯净的概率分布
        p = torch.softmax(inputs, dim=1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 3. 计算 Focal Term: (1 - pt) ^ gamma
        focal_term = (1 - pt) ** self.gamma

        # 4. 组合 Focal Loss
        loss = focal_term * ce_loss

        # 5. 手动应用类别权重 (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            batch_weights = self.alpha.gather(0, targets)
            loss = loss * batch_weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


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


def train_epoch(model, loader, optimizer, criterion, device, scaler=None, mixup_alpha=0.0,
                grad_accum_steps: int = 1, ema_model=None, swa_model=None, swa_start: int = 0, epoch: int = 0):
    """单轮训练逻辑，支持 MixUp"""
    model.train()
    if TPU_AVAILABLE and device.type == 'xla':
        freeze_batchnorm(model, is_master=(xr.global_ordinal() == 0))
    running_loss = 0.0
    correct = 0
    total = 0

    loader_wrapper = loader
    if TPU_AVAILABLE and device.type == 'xla':
        loader_wrapper = pl.ParallelLoader(
            loader, [device]).per_device_loader(device)

    optimizer.zero_grad()

    show_pbar = (not TPU_AVAILABLE) or (
        TPU_AVAILABLE and xr.global_ordinal() == 0)

    if show_pbar:
        pbar = tqdm(loader_wrapper, desc="Train", leave=False)
        iter_loader = pbar
    else:
        iter_loader = loader_wrapper
        pbar = None

    def _update_ema_swa():
        if ema_model is not None:
            ema_model.update_parameters(model)
        if (swa_model is not None) and (epoch >= swa_start):
            swa_model.update_parameters(model)

    for step, (x, y, vessel_idx, _, _) in enumerate(iter_loader):
        x, y = x.to(device), y.to(device)

        # Mixup处理
        use_mixup = (mixup_alpha > 0)

        if TPU_AVAILABLE:
            if use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha, device)
                outputs = model(x, vessel_idx)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(x, vessel_idx)
                loss = criterion(outputs, y)

            (loss / grad_accum_steps).backward()
            if (step + 1) % grad_accum_steps == 0:
                xm.optimizer_step(optimizer, barrier=True)
                xm.mark_step()
                _update_ema_swa()
                optimizer.zero_grad()
        else:
            if use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha, device)
            else:
                y_a, y_b, lam = None, None, None

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(x, vessel_idx)
                    if use_mixup:
                        loss = mixup_criterion(
                            criterion, outputs, y_a, y_b, lam)
                    else:
                        loss = criterion(outputs, y)
                scaler.scale(loss / grad_accum_steps).backward()
                if (step + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    _update_ema_swa()
                    optimizer.zero_grad()
            else:
                outputs = model(x, vessel_idx)
                if use_mixup:
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    loss = criterion(outputs, y)
                (loss / grad_accum_steps).backward()
                if (step + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    _update_ema_swa()
                    optimizer.zero_grad()

        # 统计指标
        with torch.no_grad():
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)

            if use_mixup:
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

    if TPU_AVAILABLE and (len(loader) % grad_accum_steps != 0):
        xm.optimizer_step(optimizer, barrier=True)
        xm.mark_step()
        _update_ema_swa()
        optimizer.zero_grad()
    elif (not TPU_AVAILABLE) and (len(loader) % grad_accum_steps != 0):
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        _update_ema_swa()
        optimizer.zero_grad()

    if TPU_AVAILABLE:
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

        loader_len = len(loader) * xr.world_size()
        return running_loss / (loader_len if loader_len > 0 else 1), correct / (total if total > 0 else 1)

    return running_loss / len(loader), correct / total


def validate(model, loader, device, criterion, num_classes=6):
    """验证集评估逻辑 (支持TPU多核Reduce)"""
    model.eval()
    running_loss = 0.0
    confusion_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.long, device=device)

    loader_wrapper = loader
    if TPU_AVAILABLE and device.type == 'xla':
        loader_wrapper = pl.ParallelLoader(
            loader, [device]).per_device_loader(device)

    show_pbar = (not TPU_AVAILABLE) or (
        TPU_AVAILABLE and xr.global_ordinal() == 0)
    iter_loader = tqdm(loader_wrapper, desc="Val",
                       leave=False) if show_pbar else loader_wrapper

    with torch.no_grad():
        for x, y, vessel_idx, _, _ in iter_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x, vessel_idx)
            loss = criterion(outputs, y)

            running_loss += loss.item()

            _, preds = outputs.max(1)
            indices = y * num_classes + preds
            batch_conf_mat = torch.bincount(indices, minlength=num_classes**2)
            confusion_matrix += batch_conf_mat.view(num_classes, num_classes)

    if TPU_AVAILABLE:
        if not isinstance(running_loss, torch.Tensor):
            running_loss = torch.tensor(running_loss, device=device)

        t_metrics = [running_loss, confusion_matrix]
        xm.all_reduce('sum', t_metrics)

        global_loss_sum = t_metrics[0].item()
        global_conf_mat = t_metrics[1].cpu().numpy()

        total_batches = len(loader) * xr.world_size()
        avg_loss = global_loss_sum / \
            (total_batches if total_batches > 0 else 1)

    else:
        avg_loss = running_loss / len(loader)
        global_conf_mat = confusion_matrix.cpu().numpy()

    tp_sum = np.trace(global_conf_mat)
    total_samples = np.sum(global_conf_mat)
    avg_acc = tp_sum / total_samples if total_samples > 0 else 0

    y_true_r = np.repeat(np.arange(num_classes), np.sum(
        global_conf_mat, axis=1).astype(int))
    y_pred_r_list = []
    for i in range(num_classes):
        row = global_conf_mat[i]
        for j, count in enumerate(row):
            if count > 0:
                y_pred_r_list.extend([j] * int(count))

    y_pred_r = np.array(y_pred_r_list)

    f1 = f1_score(
        y_true_r,
        y_pred_r,
        average='macro',
        labels=list(range(num_classes)),
        zero_division=0,
    ) if len(y_pred_r) > 0 else 0.0

    # 打印详细分类报告 (仅在主进程)
    if show_pbar:
        print("\nClassification Report:")
        print(classification_report(
            y_true_r,
            y_pred_r,
            labels=list(range(num_classes)),
            zero_division=0,
            digits=4
        ))

    return {
        'loss': avg_loss,
        'acc': avg_acc,
        'f1': f1
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
        for x, _, vessel_idx, rids, pids in iter_loader:
            x = x.to(device)
            outputs = model(x, vessel_idx)

            x_flip = torch.flip(x, dims=[-1])
            outputs_flip = model(x_flip, vessel_idx)
            outputs = (outputs + outputs_flip) / 2

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()

            for rid, pid, pred, prob in zip(rids, pids, preds, probs):
                row = {'ID': rid, 'Prediction': pred}
                for i, p in enumerate(prob):
                    row[f'prob_{i}'] = p
                results.append(row)

    if TPU_AVAILABLE:
        rank = xr.global_ordinal()
        part_csv = os.path.join(args.output_dir, f"part_{rank}_{output_name}")
        pd.DataFrame(results).to_csv(part_csv, index=False)

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

                for r in range(world_size):
                    fname = os.path.join(
                        args.output_dir, f"part_{r}_{output_name}")
                    if os.path.exists(fname):
                        os.remove(fname)

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
    parser.add_argument('--use_cls_token', action='store_true',
                        help='Transformer 融合时使用 CLS token 池化')
    parser.add_argument('--use_vessel_code', action='store_true',
                        help='使用 vessel_code 条件向量')
    parser.add_argument('--vessel_emb_dim', type=int, default=16,
                        help='vessel_code 嵌入维度')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--label_smoothing', type=float,
                        default=0.1, help='Label smoothing 系数')
    parser.add_argument('--focal_gamma', type=float,
                        default=2.0, help='Focal Loss gamma 参数')
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'],
                        help='损失函数类型: "ce" (CrossEntropy, 推荐 Accuracy) 或 "focal" (推荐 F1)')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='是否使用类别权重 (Accuracy 优化建议关闭)')
    parser.add_argument('--warmup_epochs', type=int,
                        default=2, help='Warmup 轮数')
    parser.add_argument('--use_ema', action='store_true', help='启用 EMA')
    parser.add_argument('--ema_decay', type=float,
                        default=0.999, help='EMA 衰减率')
    parser.add_argument('--use_swa', action='store_true', help='启用 SWA')
    parser.add_argument('--swa_start', type=int, default=10, help='SWA 开始轮次')
    parser.add_argument('--swa_lr', type=float, default=1e-5, help='SWA 学习率')
    parser.add_argument('--tpu_lr_scale', type=float, default=0.25,
                        help='TPU 学习率缩放因子 (默认1.0, 避免自动线性放大)')
    parser.add_argument('--tpu_batch_is_global', action='store_true',
                        help='TPU下将batch_size视为全局batch并自动按world_size切分')

    parser.add_argument('--patience', type=int, default=20,
                        help='Early Stopping patience')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42, help='全局随机种子')
    parser.add_argument('--mixup_alpha', type=float,
                        default=0, help='MixUp alpha 参数 (0表示禁用)')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='梯度累积步数 (用于降低显存占用)')
    parser.add_argument('--warmup_steps', type=int, default=1,
                        help='训练前 warmup 步数（用于摊薄 TPU 首轮编译成本）')

    # Jupyter kernel 兼容参数
    parser.add_argument('-f', '--file', type=str,
                        required=False, help='Jupyter kernel file (ignore)')

    args, _ = parser.parse_known_args()
    return args


def run_worker(rank, args):
    """统一的训练工作流，支持单机和TPU多进程"""
    is_tpu = (rank is not None) and TPU_AVAILABLE

    if TPU_AVAILABLE and (rank is not None):
        seed_everything(args.seed + int(rank))
    else:
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

    if is_master:
        print("正在加载数据列表...")

    train_val_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv) if os.path.exists(
        args.test_csv) else pd.DataFrame()

    for df in [train_val_df, test_df]:
        if df.empty:
            continue
        if 'pid' not in df.columns:
            col = 'check_id' if 'check_id' in df.columns else 'record_id'
            df['pid'] = df[col].astype(str).apply(lambda x: re.findall(
                r'\d+', x)[0] if re.findall(r'\d+', x) else x)

    pids = train_val_df['pid'].unique()
    train_pids, val_pids = train_test_split(
        pids, test_size=0.15, random_state=args.seed)
    train_df = train_val_df[train_val_df['pid'].isin(train_pids)].copy()
    val_df = train_val_df[train_val_df['pid'].isin(val_pids)].copy()

    test_root = args.test_img_root if os.path.exists(
        args.test_img_root) else args.img_root

    train_ds = DabangDataset(train_df, args.img_root,
                             num_views=args.num_views, augment=True)
    val_ds = DabangDataset(val_df, args.img_root,
                           num_views=args.num_views, augment=False)

    if not test_df.empty:
        test_ds = DabangDataset(
            test_df, test_root, num_views=args.num_views, is_test=True)
    else:
        test_ds = None

    loader_num_workers = 0 if is_tpu else args.num_workers
    if is_tpu and args.tpu_batch_is_global:
        per_core_batch = max(1, args.batch_size // xr.world_size())
    else:
        per_core_batch = args.batch_size
    if is_master and is_tpu:
        print(
            f"TPU per-core batch: {per_core_batch} (global={per_core_batch * xr.world_size()})")

    loader_args = {'batch_size': per_core_batch,
                   'num_workers': loader_num_workers, 'pin_memory': not is_tpu}

    if is_tpu:
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
        train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
        test_loader = DataLoader(
            test_ds, shuffle=False, **loader_args) if test_ds else []

    if is_master:
        print(f"初始化 SiameseModel ({args.model}) ...")

    model = SiameseModel(model_name=args.model,
                         num_classes=args.num_classes, num_views=args.num_views,
                         use_cls_token=args.use_cls_token,
                         use_vessel_code=args.use_vessel_code,
                         vessel_emb_dim=args.vessel_emb_dim)

    model = try_enable_sync_batchnorm(
        model, is_tpu=is_tpu, is_master=is_master)
    if is_tpu:
        model = replace_batchnorm_with_groupnorm(
            model, num_groups=16, is_master=is_master)

    if (not is_tpu) and (torch.cuda.device_count() > 1):
        if is_master:
            print(f"检测到 {torch.cuda.device_count()} 个 GPU，启用 DataParallel 并行训练")
        model = nn.DataParallel(model)

    model.to(device)

    effective_lr = args.lr * (args.tpu_lr_scale if is_tpu else 1.0)
    optimizer = optim.Adam(model.parameters(), lr=effective_lr)

    if args.use_class_weights:
        class_weights = compute_class_weights_from_df(
            train_df, args.num_classes).to(device)
        if is_master:
            print(f"已启用类别权重: {class_weights.tolist()}")
    else:
        class_weights = None
        if is_master:
            print("未启用类别权重 (优化 Accuracy)")

    # 选择损失函数
    if args.loss_type == 'focal':
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing
        )
        if is_master:
            print(f"使用 FocalLoss (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing
        )
        if is_master:
            print("使用 CrossEntropyLoss")

    scaler = torch.cuda.amp.GradScaler() if (
        not is_tpu and torch.cuda.is_available()) else None

    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    ema_model = None
    if args.use_ema:
        def _ema_avg_fn(averaged_param, param, num_averaged):
            return averaged_param * args.ema_decay + param * (1.0 - args.ema_decay)

        ema_model = AveragedModel(model, avg_fn=_ema_avg_fn).to(device)

    swa_model = None
    swa_scheduler = None
    if args.use_swa:
        swa_model = AveragedModel(model).to(device)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)

    def master_print(*args_p, **kwargs_p):
        if is_master:
            print(*args_p, **kwargs_p)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(
        args.output_dir, "best_model_f1.pt"), trace_func=master_print)

    best_acc = 0.0
    best_loss = float('inf')

    if is_master:
        print("开始训练流程...")

    # TPU warmup：运行少量 step 触发编译（不计入统计）
    if args.warmup_steps > 0:
        if is_master:
            print(f"Warmup {args.warmup_steps} step(s)...")
        model.train()
        loader_wrapper = train_loader
        if TPU_AVAILABLE and device.type == 'xla':
            loader_wrapper = pl.ParallelLoader(
                train_loader, [device]).per_device_loader(device)
        warmup_iter = iter(loader_wrapper)
        optimizer.zero_grad()
        for _ in range(args.warmup_steps):
            try:
                x, y, vessel_idx, _, _ = next(warmup_iter)
            except StopIteration:
                warmup_iter = iter(loader_wrapper)
                x, y, vessel_idx, _, _ = next(warmup_iter)
            x, y = x.to(device), y.to(device)
            outputs = model(x, vessel_idx)
            loss = criterion(outputs, y)
            loss.backward()
            if TPU_AVAILABLE and device.type == 'xla':
                xm.optimizer_step(optimizer, barrier=True)
                xm.mark_step()
            else:
                optimizer.step()
            optimizer.zero_grad()

    for epoch in range(args.epochs):
        t0 = time.time()

        if is_tpu:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            mixup_alpha=args.mixup_alpha, grad_accum_steps=args.grad_accum_steps,
            ema_model=ema_model, swa_model=swa_model, swa_start=args.swa_start, epoch=epoch)

        if args.use_swa and (swa_scheduler is not None) and epoch >= args.swa_start:
            swa_scheduler.step()
        else:
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        eval_model = ema_model if (ema_model is not None) else model
        val_metrics = validate(eval_model, val_loader, device,
                               criterion, num_classes=args.num_classes)

        dt = time.time() - t0

        if is_master:
            print(f"Epoch {epoch+1:02d} ({dt:.1f}s) lr={current_lr:.2e} | "
                  f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                  f"Val: Loss={val_metrics['loss']:.4f} Acc={val_metrics['acc']:.4f} F1={val_metrics['f1']:.4f}")

            if val_metrics['acc'] > best_acc:
                best_acc = val_metrics['acc']
                save_path = os.path.join(args.output_dir, "best_model.pt")
                if is_tpu:
                    cpu_state_dict = {k: v.cpu()
                                      for k, v in eval_model.state_dict().items()}
                    torch.save(cpu_state_dict, save_path)
                    del cpu_state_dict
                else:
                    state_dict = eval_model.module.state_dict() if isinstance(
                        eval_model, nn.DataParallel) else eval_model.state_dict()
                    torch.save(state_dict, save_path)
                print(f"  >>> [Acc] 性能提升，模型已保存至: {save_path}")

            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                save_path = os.path.join(args.output_dir, "best_model_loss.pt")
                if is_tpu:
                    cpu_state_dict = {k: v.cpu()
                                      for k, v in eval_model.state_dict().items()}
                    torch.save(cpu_state_dict, save_path)
                    del cpu_state_dict
                else:
                    state_dict = eval_model.module.state_dict() if isinstance(
                        eval_model, nn.DataParallel) else eval_model.state_dict()
                    torch.save(state_dict, save_path)
                print(f"  >>> [Loss] 性能提升，模型已保存至: {save_path}")

        early_stopping(val_metrics['f1'], eval_model)

        stop_flag = False
        if early_stopping.early_stop:
            stop_flag = True

        if is_tpu:
            flag_tensor = torch.tensor(
                1 if stop_flag else 0, dtype=torch.int, device=device)
            xm.all_reduce('max', [flag_tensor])
            stop_flag = (flag_tensor.item() > 0)

        if stop_flag:
            if is_master:
                print("Early stopping triggered!")
            break

    if args.use_swa and (swa_model is not None):
        if is_master:
            print("\n开始更新 SWA BN 统计...")
        if not is_tpu:
            update_bn(train_loader, swa_model, device=device)
        else:
            if is_master:
                print("TPU 环境下跳过 SWA 的 BN 更新")

        if is_master:
            swa_path = os.path.join(args.output_dir, "swa_model.pt")
            state_dict = swa_model.module.state_dict() if isinstance(
                swa_model, nn.DataParallel) else swa_model.state_dict()
            torch.save(state_dict, swa_path)
            print(f"SWA 模型已保存至: {swa_path}")

    if test_loader:
        if is_master:
            print("\n开始测试集预测...")

        best_acc_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(best_acc_path):
            run_inference(best_acc_path, "test_predictions_acc.csv",
                          test_loader, model, device, args)

        best_f1_path = os.path.join(args.output_dir, "best_model_f1.pt")
        if os.path.exists(best_f1_path):
            run_inference(best_f1_path, "test_predictions_f1.csv",
                          test_loader, model, device, args)

        best_loss_path = os.path.join(args.output_dir, "best_model_loss.pt")
        if os.path.exists(best_loss_path):
            run_inference(best_loss_path, "test_predictions_loss.csv",
                          test_loader, model, device, args)


def main():
    args = parse_args()

    if TPU_AVAILABLE:
        print("检测到 TPU 环境，启动多进程训练...")
        os.environ.pop('TPU_PROCESS_ADDRESSES', None)
        os.environ.pop('TPU_MESH_CONTROLLER_ADDRESS', None)
        os.environ.pop('TPU_MESH_CONTROLLER_PORT', None)

        xmp.spawn(run_worker, args=(args,), start_method='fork')
    else:
        run_worker(None, args)


if __name__ == "__main__":
    main()
