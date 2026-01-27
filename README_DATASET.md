# CAD-RADS 冠脉狭窄分级竞赛数据集说明文档

## 1. 任务背景 (Task Description)

本任务旨在利用人工智能算法，基于冠状动脉的 **CPR (曲面重建)** 图像，自动评估血管的狭窄程度（Stenosis Severity）。该评估标准参考了临床常用的 **CAD-RADS (Coronary Artery Disease - Reporting and Data System)** 分级系统。

### 核心目标
输入一个血管节段（如前降支 LGA、右冠 RCA 等）的多视角 CPR 图像（8个视角），模型需要输出该血管对应的狭窄等级（6分类）。

---

## 2. 类别定义 (Class Definition)

根据竞赛要求，狭窄程度被划分为以下 **6 个类别**。分类逻辑基于狭窄百分比 (`stenosis_percentage`)：

| 类别 ID (Class) | 狭窄程度范围 (Stenosis Range) | 临床意义 |
| :--- | :--- | :--- |
| **0** | **0%** | 无狭窄 (Normal) |
| **1** | **1% - 24%** | 轻微狭窄 (Minimal) |
| **2** | **25% - 49%** | 轻度狭窄 (Mild) |
| **3** | **50% - 69%** | 中度狭窄 (Moderate) |
| **4** | **70% - 99%** | 重度狭窄 (Severe) |
| **5** | **100%** | 完全闭塞 (Occluded) |

---

## 3. 数据集文件结构 (Dataset Structure)

数据集已按照 **病人级别 (Patient-Level)** 进行了严格划分，确保同一病人的所有血管数据只会出现在训练集或测试集中其中的一个，防止数据泄露。所有核心数据均位于 `splits` 文件夹下。

### 3.1 标注文件 (CSV)
1.  **`train_val.csv` (训练/验证集)**
    *   **用途**: 用于模型的日常训练 (Training) 和超参数验证 (Validation)。
    *   **规模**: 包含约 80% 的总数据量。
    *   **建议**: 在训练时，建议再次从这里划分出 10%-15% 作为验证集 (`val_loader`)，用于监控模型性能和早停 (Early Stopping)。

### 3.2 图像数据 (Images)
图像为单通道或三通道的 PNG/JPG 格式，每个血管节段包含 **8 个不同视角的旋转投影**。

1.  **训练图像文件夹**: `datasets/`
    *   **位置**: `\datasets\train_val\`
    *   **说明**: 该文件夹包含训练数据，在训练时，**读取 `train_val.csv` 中包含的图片**。
  
病人级别：
Train PIDs: 426
Test PIDs: 107
血管级别：
Train Records: 1193
Test Records: 302
图像级别： 
Train Images: 9544
Test Images: 2416

1.  **测试图像文件夹**: `test/`
    *   **位置**: `\datasets\test\`
    *   **说明**: 包含 `test_competition.csv` 对应的所有图像，供最终测试使用。

### 3.3 CSV 关键字段说明

| 字段名 | 说明 |
| :--- | :--- |
| `record_id` (或 `uniq_ID`) | **唯一标识符**。对应图像文件名的前缀。 |
| `pid` (或 `check_id`) | **病人 ID**。用于区分不同病患。 |
| `vessel_code` | 血管名称 (LAD: 前降支, LCX: 回旋支, RCA: 右冠状动脉)。 |
| `stenosis_class` | [标签] 原始狭窄等级 (0-5)。 |
| `stenosis_percentage` | [标签] 具体狭窄百分比数值 (0.0 - 100.0)。 |

---

## 4. 数据处理逻辑 (Preprocessing Details)

### 4.1 图像匹配逻辑
对于 CSV 中的每一行记录（即一段血管），通过 `record_id` 在图像文件夹中查找对应的 8 张多视角图像。
*   **命名模式**: `{record_id}_view01.png` 到 `{record_id}_view08.png`
*   如果图像不足 8 张，代码会自动进行填充（复制最后一张）。
*   如果图像超过 8 张，进行均匀采样。

### 4.2 数据划分策略 (Split Strategy)
本次划分采用了 **分层采样 (Stratified Sampling)** 策略：
1.  **按病人分组 (Group by Patient)**: 确保同一病人的所有数据绑定在一起。
2.  **计算病人最大风险 (Max Severity Stratification)**: 计算每个病人所有血管中最严重的狭窄等级，以此作为分层依据。
3.  这保证了测试集中既包含健康样本，也包含高危样本（如完全闭塞），其分布与训练集保持一致。

---

## 5. 提交结果格式 (Submission)

最终输出文件通常为 CSV 格式，包含以下字段：
*   `ID`: 对应测试集中的 ID。
*   `Prediction`: 模型预测的类别 (0-5)。

---

## 6. 基线方案说明 (Baseline Description)

本项目提供了一个完整的基线代码 (`baseline.py`)，包含从数据加载到生成提交文件的全流程。

### 7.1 算法思路
*   **模型架构**: 默认使用 `ResNet18`。
*   **输入处理**: 将每个样本的 **8 个视角** 图像在通道维度进行堆叠 (Stacking)。
    *   单张图像尺寸: `[1, 224, 224]`
    *   模型输入尺寸: `[8, 224, 224]` (Batch Size = N)
    *   通过设置 `in_chans=8` 调整模型第一层卷积。
*   **训练策略**:
    *   优化器: `Adam`, 学习率 `1e-4`。
    *   损失函数: `CrossEntropyLoss`。
    *   数据增强: 随机水平翻转 (`RandomHorizontalFlip`)。

### 7.2 环境依赖
请确保安装以下 Python 库：
```bash
pip install torch torchvision timm pandas numpy pillow scikit-learn tqdm
```

### 7.3 运行方式
**训练与预测**:
```bash
python baseline.py --model resnet18 --epochs 10 --batch_size 32
```
程序运行结束后，将在 `results/` 目录下生成 `test_predictions.csv` 文件，可直接用于提交。
