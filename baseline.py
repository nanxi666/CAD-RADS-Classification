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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import timm

# -------------------------------------------------------------------------
# 全局设置与工具函数
# -------------------------------------------------------------------------

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
    """自动获取可用的计算设备 (GPU/CPU)。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pct_to_class_6(pct: float) -> int:
    """
    将狭窄百分比转换为竞赛规定的 6 分类标签。
    映射规则: 0, 1-24, 25-49, 50-69, 70-99, 100
    """
    if pct <= 0: return 0
    if pct < 25: return 1
    if pct < 50: return 2
    if pct < 70: return 3
    if pct < 100: return 4
    return 5

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
        
        # 训练集数据增强: 随机水平翻转
        self.aug_transform = T.RandomHorizontalFlip(p=0.5) if augment else None

        # 预先加载数据索引
        self.samples = self._load_index(df)

    def _load_index(self, df: pd.DataFrame) -> List[Tuple]:
        """构建样本索引列表，提前处理文件查找逻辑。"""
        print(f"正在构建数据集索引 ({len(df)} 样本)...")
        samples = []
        for _, row in df.iterrows():
            rid = row.get('record_id', row.get('uniq_ID'))
            if not rid: continue
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
            if not img_files: continue
            
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

# -------------------------------------------------------------------------
# 训练与验证流程
# -------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    """单轮训练逻辑"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y, _, _ in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # 统计指标
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return running_loss / len(loader), correct / total

def validate(model, loader, device, criterion):
    """验证集评估逻辑"""
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y, _, _ in tqdm(loader, desc="Val", leave=False):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            losses.append(criterion(outputs, y).item())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return {
        'loss': np.mean(losses),
        'acc': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro')
    }

# -------------------------------------------------------------------------
# 主程序入口
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dabang Competition Baseline")
    # 数据路径配置
    parser.add_argument('--train_csv', default="./datasets/train_val.csv", help="训练集标签文件")
    parser.add_argument('--test_csv', default="./datasets/test_competition.csv", help="测试集提交文件")
    parser.add_argument('--img_root', default="./datasets/train_val", help="图像数据根目录")
    parser.add_argument('--test_img_root', default="./datasets/test", help="测试图像目录(可选)")
    parser.add_argument('--output_dir', default="./results", help="结果输出目录")
    
    # 训练参数配置
    parser.add_argument('--model', default='resnet18', help='模型架构名称 (支持 timm)')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 1. 环境初始化
    seed_everything(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"运行配置: Device={device}, Model={args.model}, Batch={args.batch_size}")
    
    # 2. 数据准备
    print("正在加载数据列表...")
    train_val_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    
    # PID 提取: 用于按病人划分验证集，防止数据泄漏
    for df in [train_val_df, test_df]:
        if 'pid' not in df.columns:
            col = 'check_id' if 'check_id' in df.columns else 'record_id'
            # 提取字符串中的数字部分作为 PID
            df['pid'] = df[col].astype(str).apply(lambda x: re.findall(r'\d+', x)[0] if re.findall(r'\d+', x) else x)

    # 划分训练集与验证集 (Ratio 85:15)
    pids = train_val_df['pid'].unique()
    train_pids, val_pids = train_test_split(pids, test_size=0.15, random_state=args.seed)
    train_df = train_val_df[train_val_df['pid'].isin(train_pids)].copy()
    val_df = train_val_df[train_val_df['pid'].isin(val_pids)].copy()
    
    # 处理测试集图像路径 (如果测试集在不同文件夹)
    test_root = args.test_img_root if os.path.exists(args.test_img_root) else args.img_root
    
    # 构建 DataLoader
    train_ds = DabangDataset(train_df, args.img_root, augment=True)
    val_ds = DabangDataset(val_df, args.img_root, augment=False)
    test_ds = DabangDataset(test_df, test_root, is_test=True)
    
    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    
    # 3. 模型与优化器初始化
    # 如果作为 Baseine 发布，这里使用最基础的定义方式
    model = BaselineModel(model_name=args.model, num_classes=6, num_views=8)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 4. 训练循环
    best_acc = 0.0
    print("开始训练流程...")
    
    for epoch in range(args.epochs):
        t0 = time.time()
        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        # 验证阶段
        val_metrics = validate(model, val_loader, device, criterion)
        
        dt = time.time() - t0
        print(f"Epoch {epoch+1:02d} ({dt:.1f}s) | "
              f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val: Loss={val_metrics['loss']:.4f} Acc={val_metrics['acc']:.4f} F1={val_metrics['f1']:.4f}")
        
        # 保存最佳模型 (以 Accuracy 为准，也可改为 F1)
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> 性能提升，模型已保存至: {save_path}")
            
    # 5. 推理生成结果
    print("\n训练完成，正在加载最佳模型进行预测...")
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    model.eval()
    results = []
    with torch.no_grad():
        for x, _, rids, pids in tqdm(test_loader, desc="Inference"):
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()
            
            for rid, pid, pred, prob in zip(rids, pids, preds, probs):
                row = {'ID': rid, 'Prediction': pred}
                # 保存每个类别的概率，便于后续融合
                for i, p in enumerate(prob):
                    row[f'prob_{i}'] = p
                results.append(row)
                
    out_csv = os.path.join(args.output_dir, "test_predictions.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"预测结果已保存至: {out_csv}")

if __name__ == "__main__":
    main()
