"""
消融实验与可视化分析脚本 (完整版)

实验设计:
    A0: YOLOv11m Baseline (无改进)
    A1: YOLOv11m + SADR (仅创新点1: 尺度自适应动态路由)
    A2: YOLOv11m + BDFR (仅创新点2: 背景解耦特征精炼)
    A3: YOLOv11m + SADR + BDFR (完整改进, 可复用已训练权重)

生成的图表 (保存到 runs/ablation/visualizations/):
    1. ablation_results.png       - 消融实验结果对比
    2. component_contribution.png - 各组件贡献分析
    3. routing_weights_vis.png    - SADR路由权重可视化分析
    4. small_target_analysis.png  - 小目标检测性能提升分析

使用方法:
    # 训练 A0/A1/A2 + 复用 A3 已训练权重 + 生成图表
    python scripts/ablation_study.py --epochs 200 --batch 32 --lr 0.0014 \\
        --a3_weights runs/road_anomaly/yolov11m_improved/weights/best.pt

    # 仅运行指定实验
    python scripts/ablation_study.py --experiments A0 A1 --epochs 200 --batch 32 --lr 0.0014

    # 仅生成图表 (所有实验已完成)
    python scripts/ablation_study.py --plot_only \\
        --a3_weights runs/road_anomaly/yolov11m_improved/weights/best.pt
"""
import os
import sys
import json
import argparse
import time
import glob
import numpy as np
from pathlib import Path
from datetime import datetime

# matplotlib 非交互后端 (服务器无显示器)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings('ignore', message='.*does not have a deterministic implementation.*')
warnings.filterwarnings('ignore', category=UserWarning)

import yaml
import torch
import torch.nn as nn

from models.register_modules import register_custom_modules
register_custom_modules()

from ultralytics import YOLO


# ================================================================
# 1. 字体与标签配置
# ================================================================
def setup_chinese_font():
    """尝试设置中文字体, 优先 SimHei 系列"""
    import matplotlib.font_manager as fm
    candidates = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Source Han Sans CN',
        'AR PL UMing CN', 'AR PL UKai CN',
    ]
    for name in candidates:
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"  ✅ 中文字体: {name}")
            return True
    # 尝试任何 CJK 字体
    for f in fm.fontManager.ttflist:
        if any(kw in f.name.lower() for kw in ['cjk', 'hei', 'song', 'kai', 'ming']):
            plt.rcParams['font.sans-serif'] = [f.name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"  ✅ CJK字体: {f.name}")
            return True
    print("  ⚠️ 未找到中文字体, 使用英文标签")
    return False


USE_CN = setup_chinese_font()


def L(cn, en):
    """双语标签选择器"""
    return cn if USE_CN else en


# ================================================================
# 2. 全局常量与实验配置
# ================================================================
ABLATION_DIR = os.path.join(PROJECT_ROOT, 'runs', 'ablation')
VIS_DIR = os.path.join(ABLATION_DIR, 'visualizations')
CLASS_NAMES = ['debris', 'illegal_parking', 'retrograde']
CLASS_NAMES_DISPLAY = [L('抛洒物', 'debris'), L('违停', 'parking'), L('逆行', 'retro')]

COLORS = {'A0': '#4C72B0', 'A1': '#55A868', 'A2': '#DD8452', 'A3': '#C44E52'}

ABLATION_CONFIGS = {
    'A0': {
        'yaml': 'models/yolov11m-baseline.yaml',
        'name': 'A0_baseline',
        'desc': 'YOLOv11m Baseline',
        'short': 'A0: Baseline',
        'sadr': False, 'bdfr': False,
    },
    'A1': {
        'yaml': 'models/yolov11m-sadr.yaml',
        'name': 'A1_sadr',
        'desc': 'YOLOv11m + SADR',
        'short': 'A1: +SADR',
        'sadr': True, 'bdfr': False,
    },
    'A2': {
        'yaml': 'models/yolov11m-bdfr.yaml',
        'name': 'A2_bdfr',
        'desc': 'YOLOv11m + BDFR',
        'short': 'A2: +BDFR',
        'sadr': False, 'bdfr': True,
    },
    'A3': {
        'yaml': 'models/yolov11m-road-anomaly.yaml',
        'name': 'A3_full',
        'desc': 'YOLOv11m + SADR + BDFR',
        'short': 'A3: Full',
        'sadr': True, 'bdfr': True,
    },
}


# ================================================================
# 3. 延迟初始化回调 (SADR/BDFR lazy build → 注入优化器)
# ================================================================
def _materialize_lazy_modules(trainer):
    """在训练循环开始前, 触发自定义模块延迟构建并将新参数注入优化器"""
    model = trainer.model
    has_lazy = any(hasattr(m, '_built') and not m._built for m in model.modules())
    if not has_lazy:
        return

    device = next(model.parameters()).device
    imgsz = trainer.args.imgsz
    params_before = {id(p) for p in model.parameters()}

    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        try:
            model(dummy)
        except Exception as e:
            print(f"  ⚠️ 延迟初始化失败: {e}")
            if was_training:
                model.train()
            return
    if was_training:
        model.train()

    g_bn, g_weight, g_bias = [], [], []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if id(param) in params_before or not param.requires_grad:
                continue
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                g_bn.append(param)
            elif name == 'bias':
                g_bias.append(param)
            else:
                g_weight.append(param)

    total_new = len(g_bn) + len(g_weight) + len(g_bias)
    if total_new == 0:
        return

    if len(trainer.optimizer.param_groups) >= 3:
        trainer.optimizer.param_groups[0]['params'].extend(g_bn)
        trainer.optimizer.param_groups[1]['params'].extend(g_weight)
        trainer.optimizer.param_groups[2]['params'].extend(g_bias)
    else:
        trainer.optimizer.add_param_group({
            'params': g_bn + g_weight + g_bias, 'lr': trainer.args.lr0
        })

    total_params = sum(p.numel() for p in g_bn + g_weight + g_bias)
    print(f"  ✅ 延迟初始化: {total_new} 组参数 ({total_params:,}) 已加入优化器")


# ================================================================
# 4. 训练与评估函数
# ================================================================
def count_parameters(model):
    """统计模型参数量(M)"""
    try:
        return sum(p.numel() for p in model.model.parameters()) / 1e6
    except Exception:
        return 0.0


def run_single_experiment(exp_id, config, args):
    """运行单个消融实验的训练"""
    print(f"\n{'=' * 60}")
    print(f"  实验 {exp_id}: {config['desc']}")
    print(f"  SADR: {'✓' if config['sadr'] else '✗'}  |  BDFR: {'✓' if config['bdfr'] else '✗'}")
    print(f"  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    print(f"{'=' * 60}")

    start_time = time.time()

    model = YOLO(config['yaml'])

    # 加载预训练权重
    pretrained = args.pretrained or 'yolo11m.pt'
    if os.path.exists(pretrained):
        model = model.load(pretrained)
        print(f"  ✅ 预训练权重: {pretrained}")

    # 含自定义模块时注册延迟初始化回调
    if config['sadr'] or config['bdfr']:
        model.add_callback("on_pretrain_routine_end", _materialize_lazy_modules)

    model.train(
        data='dataset/road_anomaly.yaml',
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer='AdamW',
        lr0=args.lr,
        lrf=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,
        momentum=0.937,
        weight_decay=0.0005,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        amp=True,
        cos_lr=True,
        close_mosaic=10,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        project=ABLATION_DIR,
        name=config['name'],
        exist_ok=True,
        plots=True,
        save_period=10,
    )

    elapsed = time.time() - start_time
    print(f"  ⏱️ {exp_id} 训练耗时: {elapsed / 60:.1f} min")
    return elapsed


def evaluate_model(best_pt, data_yaml='dataset/road_anomaly.yaml',
                   imgsz=640, device='0'):
    """评估模型, 返回指标字典"""
    model = YOLO(best_pt)
    params_m = count_parameters(model)

    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        device=device,
        plots=False,
        verbose=False,
    )

    result = {
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'params_M': round(params_m, 2),
    }

    # 各类别 AP
    try:
        per_class = {}
        for i, name in enumerate(CLASS_NAMES):
            per_class[name] = {
                'AP50': float(metrics.box.ap50[i]),
                'AP50_95': float(metrics.box.ap[i]),
            }
        result['per_class'] = per_class
    except Exception as e:
        print(f"  ⚠️ 各类别AP提取失败: {e}")

    return result


# ================================================================
# 5. 标注解析 (目标尺寸分布)
# ================================================================
def parse_val_annotations(data_yaml, imgsz=640):
    """
    解析验证集标注, 统计各类别目标尺寸分布
    尺寸标准 (像素面积, 以 imgsz×imgsz 图像计算):
        小目标: area < 32×32 = 1024
        中目标: 1024 ≤ area < 96×96 = 9216
        大目标: area ≥ 9216
    """
    with open(data_yaml, 'r') as f:
        dcfg = yaml.safe_load(f)

    base_path = dcfg.get('path', '')
    val_rel = dcfg.get('val', 'images/val')
    val_label_dir = os.path.join(base_path, val_rel.replace('images', 'labels'))

    size_stats = {name: {'small': 0, 'medium': 0, 'large': 0, 'areas': []}
                  for name in CLASS_NAMES}

    label_files = glob.glob(os.path.join(val_label_dir, '*.txt'))
    for lf in label_files:
        with open(lf, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                area = w * h * imgsz * imgsz
                if cls_id < len(CLASS_NAMES):
                    name = CLASS_NAMES[cls_id]
                    size_stats[name]['areas'].append(area)
                    if area < 1024:
                        size_stats[name]['small'] += 1
                    elif area < 9216:
                        size_stats[name]['medium'] += 1
                    else:
                        size_stats[name]['large'] += 1

    return size_stats


def find_sample_images_by_class(data_yaml, num_per_class=1):
    """为每个类别找到包含该类标注的验证集图片"""
    with open(data_yaml, 'r') as f:
        dcfg = yaml.safe_load(f)

    base_path = dcfg.get('path', '')
    val_rel = dcfg.get('val', 'images/val')
    val_img_dir = os.path.join(base_path, val_rel)
    val_label_dir = val_img_dir.replace('images', 'labels')

    selected = {}
    label_files = sorted(glob.glob(os.path.join(val_label_dir, '*.txt')))

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        found = 0
        for lf in label_files:
            with open(lf) as f:
                lines = f.readlines()
            # 确保该图中有该类别目标
            if any(line.strip().startswith(f'{cls_id} ') or
                   line.strip() == str(cls_id) for line in lines):
                img_stem = os.path.splitext(os.path.basename(lf))[0]
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG']:
                    img_path = os.path.join(val_img_dir, img_stem + ext)
                    if os.path.exists(img_path):
                        if cls_name not in selected:
                            selected[cls_name] = []
                        selected[cls_name].append(img_path)
                        found += 1
                        break
            if found >= num_per_class:
                break

    return selected


# ================================================================
# 6. 图表1: 消融实验结果对比
# ================================================================
def generate_ablation_chart(results, save_dir):
    """
    分组柱状图 + 表格: mAP50, mAP50-95, Precision, Recall
    """
    os.makedirs(save_dir, exist_ok=True)

    exp_ids = [eid for eid in ['A0', 'A1', 'A2', 'A3'] if eid in results]
    if len(exp_ids) < 2:
        print("  ⚠️ 实验数不足, 无法生成消融结果图")
        return

    metrics_keys = ['mAP50', 'mAP50_95', 'precision', 'recall']
    metric_labels = ['mAP@50', 'mAP@50-95', L('精确率(P)', 'Precision'), L('召回率(R)', 'Recall')]
    bar_colors = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']

    x = np.arange(len(exp_ids))
    width = 0.17
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (mk, ml, bc) in enumerate(zip(metrics_keys, metric_labels, bar_colors)):
        vals = [results[eid][mk] for eid in exp_ids]
        bars = ax.bar(x + offsets[i], vals, width, label=ml,
                      color=bc, edgecolor='white', linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xlabel(L('实验配置', 'Experiment'), fontsize=12)
    ax.set_ylabel(L('指标值', 'Metric Value'), fontsize=12)
    ax.set_title(L('消融实验结果对比', 'Ablation Study Results'),
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{eid}\n{ABLATION_CONFIGS[eid]['desc']}" for eid in exp_ids],
                       fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 底部数据表格
    table_data = []
    for eid in exp_ids:
        r = results[eid]
        table_data.append([
            ABLATION_CONFIGS[eid]['desc'],
            f"{r['mAP50']:.4f}", f"{r['mAP50_95']:.4f}",
            f"{r['precision']:.4f}", f"{r['recall']:.4f}",
            f"{r['params_M']:.2f}M"
        ])

    col_labels = [L('模型', 'Model'), 'mAP@50', 'mAP@50-95', 'P', 'R', L('参数量', 'Params')]
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='bottom', bbox=[0.0, -0.42, 1.0, 0.28],
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # 表头行
            cell.set_facecolor('#E8E8E8')
            cell.set_text_props(fontweight='bold')

    fig.subplots_adjust(bottom=0.32)

    path = os.path.join(save_dir, 'ablation_results.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 图1已保存: {path}")


# ================================================================
# 7. 图表2: 各组件贡献分析
# ================================================================
def generate_component_chart(results, save_dir):
    """
    左图: 组件贡献瀑布图 (Baseline → +SADR → +BDFR → Synergy → Full)
    右图: 各实验全指标对比柱状图
    """
    os.makedirs(save_dir, exist_ok=True)

    if not all(k in results for k in ['A0', 'A1', 'A2', 'A3']):
        print("  ⚠️ 缺少 A0-A3 数据, 无法生成组件贡献图")
        return

    baseline = results['A0']['mAP50']
    sadr_delta = results['A1']['mAP50'] - baseline
    bdfr_delta = results['A2']['mAP50'] - baseline
    full_val = results['A3']['mAP50']
    synergy = full_val - baseline - sadr_delta - bdfr_delta

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # ---- 左图: 瀑布图 ----
    categories = [
        L('基线(A0)', 'Baseline(A0)'),
        L('+SADR', '+SADR'),
        L('+BDFR', '+BDFR'),
        L('协同增益', 'Synergy'),
        L('完整模型(A3)', 'Full(A3)'),
    ]
    values = [baseline, sadr_delta, bdfr_delta, synergy, full_val]
    w_colors = ['#4C72B0', '#55A868', '#DD8452', '#937860', '#C44E52']

    # 累积值 (用于画连接线)
    cum = [baseline,
           baseline + sadr_delta,
           baseline + sadr_delta + bdfr_delta,
           baseline + sadr_delta + bdfr_delta + synergy]

    # 每个柱的底部
    bottoms = [0, baseline, cum[1], cum[2], 0]

    bars = ax1.bar(categories, values, bottom=bottoms, color=w_colors,
                   edgecolor='white', linewidth=1.5, width=0.6)

    # 连接虚线
    for i in range(3):
        ax1.plot([i + 0.3, i + 0.7], [cum[i], cum[i]],
                 'k--', linewidth=0.8, alpha=0.4)

    # 数值标注
    for bar, v, b in zip(bars, values, bottoms):
        if b > 0:
            label = f'+{v:.4f}' if v > 0 else f'{v:.4f}'
        else:
            label = f'{v:.4f}'
        y_pos = b + v + 0.008
        ax1.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('mAP@50', fontsize=12)
    ax1.set_title(L('组件贡献瀑布图', 'Component Contribution Waterfall'),
                  fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(full_val + 0.12, 1.0))
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    plt.setp(ax1.get_xticklabels(), rotation=15, ha='right', fontsize=9)

    # 右侧标注提升幅度
    total_improvement = full_val - baseline
    ax1.annotate(
        f'{L("总提升", "Total")}: +{total_improvement:.4f}\n'
        f'({total_improvement / baseline * 100:.1f}%↑)',
        xy=(4, full_val), xytext=(3.3, full_val + 0.06),
        fontsize=10, fontweight='bold', color='#C44E52',
        arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5),
        ha='center',
    )

    # ---- 右图: 全指标分组柱状图 ----
    metrics = ['mAP50', 'mAP50_95', 'precision', 'recall']
    metric_labels = ['mAP@50', 'mAP@50-95', 'P', 'R']
    exp_ids = ['A0', 'A1', 'A2', 'A3']

    x = np.arange(len(metrics))
    width = 0.18

    for i, eid in enumerate(exp_ids):
        vals = [results[eid][m] for m in metrics]
        ax2.bar(x + (i - 1.5) * width, vals, width,
                label=ABLATION_CONFIGS[eid]['short'],
                color=COLORS[eid], edgecolor='white', linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_labels, fontsize=10)
    ax2.set_ylabel(L('指标值', 'Metric Value'), fontsize=12)
    ax2.set_title(L('各实验指标全景对比', 'Full Metric Comparison'),
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    fig.tight_layout(w_pad=3)
    path = os.path.join(save_dir, 'component_contribution.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 图2已保存: {path}")


# ================================================================
# 8. 图表3: SADR 路由权重可视化
# ================================================================
def generate_routing_visualization(model_weights, save_dir,
                                   data_yaml='dataset/road_anomaly.yaml',
                                   imgsz=640, device='0'):
    """
    对验证集样本推理, 捕获 SADR 逐像素尺度路由权重,
    以热力图叠加原图方式展示 3 个膨胀卷积分支的激活区域.
    """
    import cv2
    from models.modules.sadr import SADR

    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    model = YOLO(model_weights)

    # 查找 SADR 模块并注册 forward hook
    captured = {}
    hooks = []
    sadr_found = False

    for name, module in model.model.named_modules():
        if isinstance(module, SADR):
            # 若未构建, 先做一次 dummy forward 触发
            if hasattr(module, '_built') and not module._built:
                print("  ℹ️ SADR 未构建, 执行 dummy forward 触发延迟初始化...")
                dev = next(model.model.parameters()).device
                dummy = torch.zeros(1, 3, imgsz, imgsz, device=dev)
                model.model.eval()
                with torch.no_grad():
                    try:
                        model.model(dummy)
                    except Exception:
                        pass

            if hasattr(module, 'scale_predictor'):
                sadr_found = True

                def make_hook(n):
                    def fn(m, inp, out):
                        captured[n] = out.detach().cpu().numpy()
                    return fn

                h = module.scale_predictor.register_forward_hook(make_hook(name))
                hooks.append(h)

    if not sadr_found:
        print("  ❌ 未找到 SADR scale_predictor, 跳过路由权重可视化")
        return

    # 为每个类别找样本图片
    selected = find_sample_images_by_class(data_yaml, num_per_class=1)
    if not selected:
        print("  ⚠️ 未找到合适的样本图片, 跳过路由权重可视化")
        for h in hooks:
            h.remove()
        return

    n_rows = len(selected)
    branch_titles = [
        L('分支 d=1\n(小感受野 3×3)', 'Branch d=1\n(Small RF 3×3)'),
        L('分支 d=3\n(中感受野 7×7)', 'Branch d=3\n(Med RF 7×7)'),
        L('分支 d=5\n(大感受野 11×11)', 'Branch d=5\n(Large RF 11×11)'),
    ]

    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (cls_name, img_list) in enumerate(selected.items()):
        img_path = img_list[0]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = img_rgb.shape[:2]

        # 推理, 触发 hook 捕获路由权重
        captured.clear()
        pred_results = model.predict(source=img_path, imgsz=imgsz,
                                     device=device, verbose=False)

        # 原图 + 检测框
        axes[row, 0].imshow(img_rgb)
        # 绘制检测框
        if pred_results and len(pred_results[0].boxes):
            for box in pred_results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                color = ['red', 'lime', 'cyan'][cls_id % 3]
                rect = plt.Rectangle((xyxy[0], xyxy[1]),
                                     xyxy[2] - xyxy[0], xyxy[3] - xyxy[1],
                                     linewidth=2, edgecolor=color, facecolor='none')
                axes[row, 0].add_patch(rect)
                axes[row, 0].text(xyxy[0], xyxy[1] - 5,
                                  f'{CLASS_NAMES[cls_id]} {conf:.2f}',
                                  color=color, fontsize=7,
                                  bbox=dict(boxstyle='round,pad=0.2',
                                            facecolor='black', alpha=0.6))
        cls_display = CLASS_NAMES_DISPLAY[CLASS_NAMES.index(cls_name)]
        axes[row, 0].set_title(f'{L("原图", "Original")} ({cls_display})', fontsize=11)
        axes[row, 0].axis('off')

        # 获取路由权重
        if not captured:
            for k in range(3):
                axes[row, k + 1].text(0.5, 0.5, 'No data', ha='center',
                                      va='center', transform=axes[row, k + 1].transAxes)
                axes[row, k + 1].axis('off')
            continue

        routing_w = list(captured.values())[0]  # (B, num_branches, H, W)
        routing_w = routing_w[0]  # (num_branches, H, W)

        for k in range(min(3, routing_w.shape[0])):
            w_k = routing_w[k]  # (H_feat, W_feat) e.g. (20, 20)

            # 上采样到原图尺寸
            w_resized = cv2.resize(w_k, (w_img, h_img),
                                   interpolation=cv2.INTER_LINEAR)

            # 热力图叠加
            heatmap = plt.cm.jet(w_resized)[:, :, :3]
            overlay = np.clip(0.4 * (img_rgb / 255.0) + 0.6 * heatmap, 0, 1)

            axes[row, k + 1].imshow(overlay)
            axes[row, k + 1].set_title(branch_titles[k], fontsize=9)
            axes[row, k + 1].axis('off')

            # 添加颜色条
            vmin, vmax = w_resized.min(), w_resized.max()
            axes[row, k + 1].text(
                0.98, 0.02,
                f'[{vmin:.2f}, {vmax:.2f}]',
                transform=axes[row, k + 1].transAxes,
                ha='right', va='bottom', fontsize=7,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    fig.suptitle(L('SADR 路由权重可视化分析\n'
                    '(暖色=高权重, 表示该尺度分支在此区域起主导作用)',
                    'SADR Routing Weight Visualization\n'
                    '(Warm=High weight, branch dominates at that region)'),
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(save_dir, 'routing_weights_vis.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 图3已保存: {path}")

    # 清理 hooks
    for h in hooks:
        h.remove()


# ================================================================
# 9. 图表4: 小目标检测性能提升分析
# ================================================================
def generate_small_target_chart(results, save_dir,
                                data_yaml='dataset/road_anomaly.yaml',
                                imgsz=640):
    """
    左图: 验证集各类别目标尺寸分布 (小/中/大)
    右图: 各类别 AP@50 对比, 标注 debris (小目标) 的提升幅度
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # ---- 左图: 目标尺寸分布 ----
    try:
        size_stats = parse_val_annotations(data_yaml, imgsz=imgsz)

        x = np.arange(len(CLASS_NAMES))
        width = 0.22

        small_c = [size_stats[c]['small'] for c in CLASS_NAMES]
        medium_c = [size_stats[c]['medium'] for c in CLASS_NAMES]
        large_c = [size_stats[c]['large'] for c in CLASS_NAMES]

        b1 = ax1.bar(x - width, small_c, width,
                     label=L('小目标 (<32²px)', 'Small (<32²px)'),
                     color='#FF6B6B', edgecolor='white')
        b2 = ax1.bar(x, medium_c, width,
                     label=L('中目标 (32²~96²px)', 'Medium (32²~96²px)'),
                     color='#4ECDC4', edgecolor='white')
        b3 = ax1.bar(x + width, large_c, width,
                     label=L('大目标 (≥96²px)', 'Large (≥96²px)'),
                     color='#45B7D1', edgecolor='white')

        # 数值标注
        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax1.text(bar.get_x() + bar.get_width() / 2, h + 1,
                             str(int(h)), ha='center', fontsize=8)

        # 标注 debris 的小目标占比
        total_debris = sum([small_c[0], medium_c[0], large_c[0]])
        if total_debris > 0:
            pct = small_c[0] / total_debris * 100
            ax1.annotate(
                f'{L("小目标占比", "Small ratio")}: {pct:.0f}%',
                xy=(0 - width, small_c[0]),
                xytext=(-0.5, max(small_c[0], medium_c[0], large_c[0]) + 15),
                fontsize=10, fontweight='bold', color='#FF6B6B',
                arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.5),
                ha='center',
            )

        ax1.set_xticks(x)
        ax1.set_xticklabels(CLASS_NAMES_DISPLAY, fontsize=10)
        ax1.set_ylabel(L('目标数量', 'Count'), fontsize=12)
        ax1.set_title(L('验证集目标尺寸分布', 'Val Set Target Size Distribution'),
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

    except Exception as e:
        ax1.text(0.5, 0.5, f'Size analysis failed:\n{e}',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=10)
        print(f"  ⚠️ 目标尺寸分析失败: {e}")

    # ---- 右图: 各类别 AP@50 对比 ----
    exp_ids = [eid for eid in ['A0', 'A1', 'A2', 'A3']
               if eid in results and 'per_class' in results[eid]]

    if len(exp_ids) >= 2:
        x = np.arange(len(CLASS_NAMES))
        width = 0.18
        n_exp = len(exp_ids)

        for i, eid in enumerate(exp_ids):
            ap50s = [results[eid]['per_class'][c]['AP50'] for c in CLASS_NAMES]
            offset = (i - (n_exp - 1) / 2) * width
            bars = ax2.bar(x + offset, ap50s, width,
                           label=ABLATION_CONFIGS[eid]['short'],
                           color=COLORS.get(eid, f'C{i}'),
                           edgecolor='white', linewidth=0.5)
            for bar, v in zip(bars, ap50s):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.008,
                         f'{v:.3f}', ha='center', va='bottom',
                         fontsize=7, rotation=45)

        # 标注 debris 的提升幅度 (A0 → A3)
        if 'A0' in results and 'A3' in results:
            a0_d = results['A0']['per_class']['debris']['AP50']
            a3_d = results['A3']['per_class']['debris']['AP50']
            delta = a3_d - a0_d
            if a0_d > 0:
                pct = delta / a0_d * 100
                ax2.annotate(
                    f'Δ = +{delta:.3f}\n({pct:.1f}%↑)',
                    xy=(0 + (n_exp - 1) / 2 * width, a3_d),
                    xytext=(0.6, min(a3_d + 0.12, 0.95)),
                    fontsize=11, fontweight='bold', color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3CD',
                              edgecolor='red', alpha=0.9),
                )

        ax2.set_xticks(x)
        ax2.set_xticklabels(CLASS_NAMES_DISPLAY, fontsize=10)
        ax2.set_ylabel('AP@50', fontsize=12)
        ax2.set_title(L('各类别 AP@50 对比 (小目标提升分析)',
                        'Per-class AP@50 (Small Target Improvement)'),
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9, loc='upper right')
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    else:
        ax2.text(0.5, 0.5, 'Insufficient experiment data',
                 ha='center', va='center', transform=ax2.transAxes)

    fig.tight_layout(w_pad=3)
    path = os.path.join(save_dir, 'small_target_analysis.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 图4已保存: {path}")


# ================================================================
# 10. 主函数
# ================================================================
# A3 (完整改进模型) 已训练权重的自动搜索路径列表
A3_WEIGHT_CANDIDATES = [
    'runs/road_anomaly/yolov11m_improved/weights/best.pt',
    'runs/ablation/A3_full/weights/best.pt',
]


def auto_detect_a3_weights():
    """自动搜索 A3 已训练权重, 避免重复训练"""
    for candidate in A3_WEIGHT_CANDIDATES:
        # 支持相对路径和绝对路径
        path = candidate if os.path.isabs(candidate) else os.path.join(PROJECT_ROOT, candidate)
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description='消融实验与可视化分析',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--experiments', nargs='+', default=None,
                        choices=['A0', 'A1', 'A2', 'A3'],
                        help='指定训练的实验 (默认 A0 A1 A2, A3 自动复用已有权重)')
    parser.add_argument('--a3_weights', type=str, default=None,
                        help='A3 已训练权重路径 (默认自动搜索)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.0014)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--plot_only', action='store_true',
                        help='跳过训练, 仅从已有权重生成图表')
    parser.add_argument('--no_skip', action='store_true',
                        help='强制重新训练所有实验 (默认会跳过已有权重的实验)')

    args = parser.parse_args()

    # 默认启用 skip_existing (除非用户传了 --no_skip)
    args.skip_existing = not args.no_skip

    # 自动搜索 A3 权重
    if args.a3_weights is None:
        args.a3_weights = auto_detect_a3_weights()
        if args.a3_weights:
            print(f"  🔍 自动发现 A3 权重: {args.a3_weights}")

    os.chdir(PROJECT_ROOT)

    print("\n" + "=" * 70)
    print("  消融实验 - YOLOv11m 道路异常事件检测")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  epochs={args.epochs}  batch={args.batch}  lr={args.lr}  imgsz={args.imgsz}")
    if args.a3_weights:
        print(f"  A3 复用权重: {args.a3_weights}")
    print("=" * 70)

    # 确定要训练的实验
    if args.experiments:
        train_exps = list(args.experiments)
    else:
        train_exps = ['A0', 'A1', 'A2']

    # A3 有现成权重则不训练; 无权重时加入训练队列
    if args.a3_weights:
        train_exps = [e for e in train_exps if e != 'A3']
    elif 'A3' not in train_exps:
        # 用户未指定 --experiments 且无已有 A3 权重, 加入训练
        train_exps.append('A3')

    # =========================================================
    # Phase 1: 训练
    # =========================================================
    if not args.plot_only:
        # 统计实际需训练的数量
        need_train = []
        for exp_id in train_exps:
            config = ABLATION_CONFIGS[exp_id]
            bp = os.path.join(ABLATION_DIR, config['name'], 'weights', 'best.pt')
            if args.skip_existing and os.path.exists(bp):
                need_train.append((exp_id, True))   # True = skip
            else:
                need_train.append((exp_id, False))  # False = train

        n_skip = sum(1 for _, s in need_train if s)
        n_run = sum(1 for _, s in need_train if not s)

        print(f"\n{'=' * 70}")
        print(f"  Phase 1: 消融实验训练")
        print(f"    待训练: {n_run} 个  |  已有权重跳过: {n_skip} 个")
        print(f"{'=' * 70}")

        for exp_id, skip in need_train:
            config = ABLATION_CONFIGS[exp_id]
            if skip:
                bp = os.path.join(ABLATION_DIR, config['name'], 'weights', 'best.pt')
                print(f"\n  ⏭️ {exp_id} ({config['desc']}) 已有权重, 跳过: {bp}")
                continue

            run_single_experiment(exp_id, config, args)

    # =========================================================
    # Phase 2: 评估所有实验
    # =========================================================
    print(f"\n{'=' * 70}")
    print(f"  Phase 2: 评估所有模型")
    print(f"{'=' * 70}")

    results_all = {}

    for exp_id, config in ABLATION_CONFIGS.items():
        # 确定权重路径
        if exp_id == 'A3' and args.a3_weights:
            wp = args.a3_weights
            if not os.path.isabs(wp):
                wp = os.path.join(PROJECT_ROOT, wp)
        else:
            wp = os.path.join(ABLATION_DIR, config['name'], 'weights', 'best.pt')

        if not os.path.exists(wp):
            print(f"  ⚠️ {exp_id} 权重不存在: {wp}, 跳过")
            continue

        print(f"\n  评估 {exp_id}: {config['desc']}")
        print(f"    权重: {wp}")
        result = evaluate_model(wp, imgsz=args.imgsz, device=args.device)
        result['name'] = config['desc']
        result['sadr'] = config['sadr']
        result['bdfr'] = config['bdfr']
        results_all[exp_id] = result

    # 保存结果 JSON
    os.makedirs(ABLATION_DIR, exist_ok=True)
    results_file = os.path.join(ABLATION_DIR, 'ablation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 结果JSON: {results_file}")

    # 打印汇总表
    print(f"\n{'=' * 100}")
    print("  消融实验结果汇总")
    print(f"{'=' * 100}")
    header = (f"  {'实验':<6} {'模型':<28} {'SADR':^5} {'BDFR':^5} "
              f"{'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} {'Params':>8}")
    print(header)
    print(f"  {'-' * 92}")

    for eid in ['A0', 'A1', 'A2', 'A3']:
        if eid not in results_all:
            continue
        r = results_all[eid]
        sadr = '✓' if r['sadr'] else '✗'
        bdfr = '✓' if r['bdfr'] else '✗'
        print(f"  {eid:<6} {r['name']:<28} {sadr:^5} {bdfr:^5} "
              f"{r['mAP50']:>8.4f} {r['mAP50_95']:>10.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['params_M']:>7.2f}M")

    # 各类别详情
    if any('per_class' in results_all[e] for e in results_all):
        print(f"\n  {'─' * 92}")
        print("  各类别 AP@50 详情:")
        print(f"  {'实验':<6}", end='')
        for cn in CLASS_NAMES:
            print(f"  {cn:>17}", end='')
        print()
        for eid in ['A0', 'A1', 'A2', 'A3']:
            if eid not in results_all or 'per_class' not in results_all[eid]:
                continue
            print(f"  {eid:<6}", end='')
            for cn in CLASS_NAMES:
                ap = results_all[eid]['per_class'][cn]['AP50']
                print(f"  {ap:>17.4f}", end='')
            print()

    # 提升统计
    if 'A0' in results_all and 'A3' in results_all:
        delta = results_all['A3']['mAP50'] - results_all['A0']['mAP50']
        pct = delta / results_all['A0']['mAP50'] * 100 if results_all['A0']['mAP50'] > 0 else 0
        print(f"\n  📈 A3 vs A0 mAP@50 提升: +{delta:.4f} ({pct:.1f}%)")

    print(f"{'=' * 100}")

    # =========================================================
    # Phase 3: 生成可视化图表
    # =========================================================
    print(f"\n{'=' * 70}")
    print(f"  Phase 3: 生成可视化图表")
    print(f"{'=' * 70}")

    # 图1: 消融实验结果对比
    print("\n  [1/4] 生成消融实验结果对比图...")
    generate_ablation_chart(results_all, VIS_DIR)

    # 图2: 各组件贡献分析
    print("\n  [2/4] 生成各组件贡献分析图...")
    generate_component_chart(results_all, VIS_DIR)

    # 图3: SADR 路由权重可视化
    print("\n  [3/4] 生成SADR路由权重可视化...")
    # 优先用 A3 权重 (完整模型含SADR), 其次 A1
    routing_weights = None
    if args.a3_weights:
        routing_weights = args.a3_weights
        if not os.path.isabs(routing_weights):
            routing_weights = os.path.join(PROJECT_ROOT, routing_weights)
    else:
        for candidate in ['A3', 'A1']:
            if candidate in ABLATION_CONFIGS:
                wp = os.path.join(ABLATION_DIR,
                                  ABLATION_CONFIGS[candidate]['name'],
                                  'weights', 'best.pt')
                if os.path.exists(wp):
                    routing_weights = wp
                    break

    if routing_weights and os.path.exists(routing_weights):
        generate_routing_visualization(
            routing_weights, VIS_DIR,
            data_yaml='dataset/road_anomaly.yaml',
            imgsz=args.imgsz, device=args.device)
    else:
        print("  ⚠️ 无可用的含 SADR 模型权重, 跳过路由权重可视化")

    # 图4: 小目标检测性能提升分析
    print("\n  [4/4] 生成小目标检测性能提升分析图...")
    generate_small_target_chart(results_all, VIS_DIR,
                                data_yaml='dataset/road_anomaly.yaml',
                                imgsz=args.imgsz)

    # =========================================================
    # 完成
    # =========================================================
    print(f"\n{'=' * 70}")
    print(f"  ✅ 消融实验全部完成!")
    print(f"  📁 结果目录: {ABLATION_DIR}")
    print(f"  📊 图表目录: {VIS_DIR}")
    print(f"  📄 结果文件: {results_file}")
    print(f"  ⏱️ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
