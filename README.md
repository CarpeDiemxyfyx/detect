# 基于改进YOLOv11的多类型道路异常事件检测系统

## 📋 项目概述

本项目实现了一个基于改进YOLOv11的道路异常事件检测系统，能同时检测三类异常事件：
- 🚧 **抛洒物** (路面散落物体)
- 🚗 **机动车违停** (违规停放车辆)
- 🔄 **逆行** (车辆逆向行驶)

系统采用 **"帧级检测 → 视频级判定"** 的两阶段架构，不仅能逐帧检测异常目标，还能通过 TVAD 模块将逐帧检测结果聚合为视频级事件决策。

### 🌟 核心创新

1. **SADR (尺度自适应动态路由模块)** - 逐像素预测尺度亲和力权重，动态路由到不同膨胀率的卷积分支
2. **BDFR (背景解耦特征精炼模块)** - 学习背景原型向量，基于偏离度生成注意力增强异常前景
3. **TVAD (时序感知视频聚合决策模块)** - 三维度评分 S_c = r_c × τ_c × conf_c，实现帧级检测到视频级事件判定的升维

### 🏗 系统架构

```
视频输入 → 逐帧检测 (YOLOv11+SADR+BDFR) → 帧级检测结果
                                              ↓
                                    TVAD 聚合决策模块
                                     ├─ 帧占比 r_c
                                     ├─ 时序一致性 τ_c (滑窗分析)
                                     └─ 平均置信度 conf_c
                                              ↓
                                    视频级事件判定结果
```

## 📁 项目结构

```
抛洒物检测/
├── data/                          # 原始数据 (视频)
│   ├── 抛洒物/                    # ~180个视频
│   ├── 机动车违停/                # ~100个视频
│   └── 逆行/                     # ~50个视频
│
├── dataset/                       # 处理后数据集
│   ├── road_anomaly.yaml         # 数据集配置
│   ├── images/{train,val,test}/  # 图片
│   └── labels/{train,val,test}/  # 标签
│
├── models/                        # 模型定义
│   ├── modules/
│   │   ├── sadr.py               # ★ 创新点1: SADR模块
│   │   ├── bdfr.py               # ★ 创新点2: BDFR模块
│   │   └── tvad.py               # ★ 创新点3: TVAD模块
│   ├── yolov11-road-anomaly.yaml # 完整改进模型配置
│   ├── yolov11m-baseline.yaml    # 基线模型 (消融A0)
│   ├── yolov11m-sadr.yaml        # 仅SADR (消融A1)
│   ├── yolov11m-bdfr.yaml        # 仅BDFR (消融A2)
│   └── register_modules.py       # 自定义模块注册
│
├── scripts/                       # 脚本
│   ├── extract_frames.py         # 视频抽帧 (按类别自适应FPS)
│   ├── split_dataset.py          # 数据集划分 (按视频组, 防泄露)
│   ├── train.py                  # 训练脚本
│   ├── inference.py              # 推理 (帧级 + 视频级TVAD)
│   ├── evaluate.py               # 评估 (帧级mAP + 视频级V-Acc/V-F1)
│   ├── ablation_study.py         # 消融实验 (A0~A4)
│   ├── comparison_experiment.py  # 对比实验
│   ├── visualize.py              # 可视化 (含TVAD时间线)
│   ├── augment_data.py           # 数据增强
│   └── auto_label_assist.py      # 半自动标注辅助
│
├── app/                           # PyQt5 应用
│   └── main_ui.py                # 检测系统界面 (含视频级判定)
│
├── runs/                          # 实验结果 (训练后生成)
├── requirements.txt              # Python依赖
└── README.md                     # 本文件
```

## 🚀 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# Step 1: 从视频中抽帧 (自适应FPS: 抛洒物=6, 违停=4, 逆行=8)
python scripts/extract_frames.py --data_dir data --output_dir dataset/images/all

# Step 2: 使用 LabelImg 标注 (YOLO格式)
#   类别: 0=debris, 1=illegal_parking, 2=retrograde
#   标签保存到: dataset/labels/all/

# Step 3: 按视频组划分数据集 (7:2:1, 防止数据泄露)
python scripts/split_dataset.py --source dataset/images/all --labels dataset/labels/all
```

### 3. 模型训练

```bash
# 训练完整改进模型
python scripts/train.py --config full --epochs 200 --batch 16

# 训练基线模型
python scripts/train.py --config baseline --epochs 200

# 断点续训
python scripts/train.py --resume runs/road_anomaly/yolov11m_improved/weights/last.pt
```

### 4. 消融实验 (A0~A4)

```bash
# 运行全部5组消融实验
python scripts/ablation_study.py --epochs 200

# 仅运行指定实验
python scripts/ablation_study.py --experiments A0 A3 A4
```

### 5. 对比实验

```bash
python scripts/comparison_experiment.py --epochs 200
```

### 6. 模型评估

```bash
# 帧级评估
python scripts/evaluate.py --weights best.pt --speed_test --visualize

# 帧级 + 视频级评估 (TVAD)
python scripts/evaluate.py --weights best.pt --video_eval --test_video_dir data
```

### 7. 推理

```bash
# 帧级推理
python scripts/inference.py --source test.mp4 --weights best.pt

# 视频级推理 (TVAD 三维度聚合判定)
python scripts/inference.py --source test.mp4 --weights best.pt --video

# 批量视频推理
python scripts/inference.py --source video_dir/ --weights best.pt --video

# 自定义 TVAD 参数
python scripts/inference.py --source test.mp4 --video --temporal_window 3.0 --suppression_alpha 0.4

# 导出 ONNX
python scripts/inference.py --export onnx --weights best.pt
```

### 8. 可视化

```bash
# 消融实验柱状图
python scripts/visualize.py --mode ablation

# TVAD 帧级/视频级指标对比图
python scripts/visualize.py --mode tvad --results_file runs/ablation/ablation_results.json

# 时序一致性时间线
python scripts/visualize.py --mode timeline --video_report runs/inference/video_report.json

# 训练曲线对比
python scripts/visualize.py --mode curves --results_dir runs/ablation
```

### 9. 启动检测系统

```bash
python app/main_ui.py --weights runs/road_anomaly/yolov11m_improved/weights/best.pt
```

## 📊 实验设计

### 消融实验

| 实验 | SADR | BDFR | TVAD | 帧级指标 | 视频级指标 |
|------|:----:|:----:|:----:|----------|------------|
| A0 | ✗ | ✗ | ✗ | mAP50, P, R | -- |
| A1 | ✓ | ✗ | ✗ | mAP50, P, R | -- |
| A2 | ✗ | ✓ | ✗ | mAP50, P, R | -- |
| A3 | ✓ | ✓ | ✗ | mAP50, P, R | -- |
| A4 | ✓ | ✓ | ✓ | mAP50, P, R | V-Acc, V-F1 |

### TVAD 三维度评分

$$S_c = r_c \times \tau_c \times conf_c$$

| 维度 | 含义 | 计算方式 |
|------|------|----------|
| $r_c$ | 帧占比 | 检出帧数 / 总帧数 |
| $\tau_c$ | 时序一致性 | 滑窗内连续检出比率的均值 |
| $conf_c$ | 平均置信度 | 该类别所有检出框置信度均值 |

### 对比实验

| 模型 | 类型 | 视频级评估 |
|------|------|:----------:|
| YOLOv5m | 经典YOLO | ✗ |
| YOLOv8m | 前代YOLO | ✗ |
| YOLOv11m | 当代基线 | ✗ |
| RT-DETR-l | Transformer | ✗ |
| **Ours** | 改进YOLOv11m | ✓ (V-Acc, V-F1) |

## 🏗 技术架构详解

```
输入 → Backbone (YOLOv11 + SADR) → Neck (FPN+PAN + BDFR) → Detect Head → 三类输出
                                                                              ↓
                                                                    TVAD 视频级聚合
```

- **SADR** 嵌入 Backbone 末端 (SPPF之后): 逐像素自适应选择感受野
- **BDFR** 嵌入 Neck 各融合节点: 基于背景原型偏离度精炼前景特征
- **TVAD** 作为推理阶段后处理: 将逐帧检测聚合为视频级事件决策

## 📄 License

本项目仅用于学术研究。
