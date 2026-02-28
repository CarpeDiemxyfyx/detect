"""
可视化工具
提供偏离度图可视化、检测结果对比等论文用图生成功能

功能:
- BDFR 偏离度热力图可视化
- 检测结果对比图 (Baseline vs Ours)
- 各类别检测效果展示
- 训练曲线对比图
- 混淆矩阵美化
- TVAD 实验可视化 (V0-V4 柱状图)
- 时序一致性时间线可视化

使用方法:
    python scripts/visualize.py --mode deviation --image test.jpg --weights best.pt
    python scripts/visualize.py --mode compare --results_dir runs/ablation
    python scripts/visualize.py --mode tvad --results_file runs/ablation/ablation_results.json
    python scripts/visualize.py --mode timeline --video_report runs/inference/video_report.json
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def visualize_deviation_map(image_path: str, weights_path: str, output_dir: str):
    """
    可视化 BDFR 模块的偏离度图
    展示模型认为哪些区域是异常前景 (高偏离度)
    """
    import cv2
    import torch
    from models.register_modules import register_custom_modules
    register_custom_modules()
    from ultralytics import YOLO
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(weights_path)
    
    # 推理
    results = model.predict(source=image_path, conf=0.3, save=False, verbose=False)
    
    # 读取原图用于叠加
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] 无法读取图片: {image_path}")
        return
    
    # 保存检测结果
    for r in results:
        annotated = r.plot()
        fname = Path(image_path).stem
        cv2.imwrite(os.path.join(output_dir, f'{fname}_detection.jpg'), annotated)
    
    print(f"  可视化结果已保存到: {output_dir}")


def plot_training_curves(results_dir: str, output_dir: str):
    """
    绘制训练曲线对比图 (loss, mAP, precision, recall)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[!] 需要安装 matplotlib 和 pandas: pip install matplotlib pandas")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有 results.csv 文件
    csv_files = list(Path(results_dir).rglob('results.csv'))
    
    if not csv_files:
        print(f"[!] 在 {results_dir} 中未找到 results.csv 文件")
        return
    
    # 要绘制的指标
    metrics_to_plot = {
        'train/box_loss': '边框损失 (Box Loss)',
        'train/cls_loss': '分类损失 (Cls Loss)',
        'metrics/mAP50(B)': 'mAP@0.5',
        'metrics/mAP50-95(B)': 'mAP@0.5:0.95',
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)': 'Recall',
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for csv_path in csv_files:
        exp_name = csv_path.parent.name
        df = pd.read_csv(csv_path)
        # 清理列名
        df.columns = df.columns.str.strip()
        
        color = colors[csv_files.index(csv_path) % len(colors)]
        
        for idx, (col, title) in enumerate(metrics_to_plot.items()):
            if col in df.columns and idx < len(axes):
                axes[idx].plot(df['epoch'], df[col], label=exp_name, 
                             color=color, linewidth=1.5)
                axes[idx].set_title(title, fontsize=12)
                axes[idx].set_xlabel('Epoch')
                axes[idx].legend(fontsize=8)
                axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  训练曲线对比图已保存: {save_path}")


def plot_ablation_bar_chart(results_file: str, output_dir: str):
    """
    绘制消融实验柱状图
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import json
    except ImportError:
        print("[!] 需要安装 matplotlib")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    exp_names = list(results.keys())
    mAP50 = [results[k].get('mAP50', 0) * 100 for k in exp_names]
    mAP50_95 = [results[k].get('mAP50_95', 0) * 100 for k in exp_names]
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, mAP50, width, label='mAP@0.5', 
                   color='#3498db', alpha=0.85)
    bars2 = ax.bar(x + width/2, mAP50_95, width, label='mAP@0.5:0.95', 
                   color='#e74c3c', alpha=0.85)
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('mAP (%)', fontsize=12)
    ax.set_title('Ablation Study Results', fontsize=14)
    ax.set_xticks(x)
    
    labels = [results[k].get('name', k) for k in exp_names]
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ablation_bar_chart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  消融实验柱状图已保存: {save_path}")


def plot_tvad_experiment(results_file: str, output_dir: str):
    """
    绘制 TVAD 实验可视化 (V0-V4 参数敏感性分析柱状图)
    
    如果消融结果中包含 V-Acc/V-F1, 在同一张图中绘制帧级和视频级指标对比
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import json
    except ImportError:
        print("[!] 需要安装 matplotlib")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 筛选有 V-Acc 的实验
    tvad_exps = {k: v for k, v in results.items() 
                 if 'V_Acc' in v or 'V_F1' in v}
    
    if not tvad_exps:
        # 没有 TVAD 实验数据, 绘制帧级 vs 视频级对比图 (A0-A4)
        print("  [提示] 未找到 TVAD 实验数据, 绘制消融实验帧级/视频级对比图")
    
    # 绘制消融实验 + TVAD 综合对比
    exp_names = list(results.keys())
    mAP50 = [results[k].get('mAP50', 0) * 100 for k in exp_names]
    v_acc = [results[k].get('V_Acc', 0) * 100 for k in exp_names]
    v_f1 = [results[k].get('V_F1', 0) * 100 for k in exp_names]
    
    x = np.arange(len(exp_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, mAP50, width, label='mAP@0.5 (Frame)',
                   color='#3498db', alpha=0.85)
    bars2 = ax.bar(x, v_acc, width, label='V-Acc (Video)',
                   color='#2ecc71', alpha=0.85)
    bars3 = ax.bar(x + width, v_f1, width, label='V-F1 (Video)',
                   color='#e74c3c', alpha=0.85)
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Ablation: Frame-level vs Video-level Metrics', fontsize=14)
    ax.set_xticks(x)
    labels = [results[k].get('name', k) for k in exp_names]
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'tvad_experiment_chart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  TVAD 实验对比图已保存: {save_path}")


def plot_temporal_timeline(video_report_file: str, output_dir: str):
    """
    绘制视频级推理的时序一致性时间线图
    展示各类别检出在时间轴上的分布
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import json
    except ImportError:
        print("[!] 需要安装 matplotlib")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(video_report_file, 'r', encoding='utf-8') as f:
        reports = json.load(f)
    
    CLASS_COLORS = {
        '0': '#e74c3c',   # 抛洒物
        '1': '#f39c12',   # 违停
        '2': '#3498db',   # 逆行
    }
    CLASS_NAMES_CN = {'0': '抛洒物', '1': '机动车违停', '2': '逆行'}
    
    for report in reports[:10]:  # 最多绘制前10个视频
        vname = report.get('video', 'unknown')
        timeline = report.get('timeline', {})
        duration = report.get('duration_sec', 0)
        primary = report.get('primary_event', {})
        
        if not timeline:
            continue
        
        # 收集所有秒级数据
        all_secs = sorted([int(s) for s in timeline.keys()])
        all_classes = set()
        for sec_data in timeline.values():
            all_classes.update(sec_data.keys())
        all_classes = sorted(all_classes)
        
        fig, ax = plt.subplots(figsize=(max(12, duration * 0.3), 4))
        
        for ci, cls_id in enumerate(all_classes):
            secs = []
            counts = []
            for s in all_secs:
                cnt = timeline.get(str(s), {}).get(cls_id, 0)
                if cnt > 0:
                    secs.append(s)
                    counts.append(cnt)
            
            color = CLASS_COLORS.get(cls_id, '#95a5a6')
            label = CLASS_NAMES_CN.get(cls_id, f'class_{cls_id}')
            ax.bar(secs, counts, width=0.8, label=label,
                   color=color, alpha=0.7, bottom=[0]*len(secs))
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Detections', fontsize=11)
        
        primary_name = primary.get('name_cn', '无') if primary else '无'
        ax.set_title(f'{vname} — 主事件: {primary_name}', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        safe_name = Path(vname).stem
        save_path = os.path.join(output_dir, f'timeline_{safe_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  时间线图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='可视化工具')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['deviation', 'curves', 'ablation', 'tvad', 'timeline'],
                       help='可视化模式')
    parser.add_argument('--image', type=str, default=None,
                       help='输入图片路径 (deviation模式)')
    parser.add_argument('--weights', type=str, default=None,
                       help='模型权重路径')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='结果目录')
    parser.add_argument('--results_file', type=str, default=None,
                       help='结果JSON文件')
    parser.add_argument('--video_report', type=str, default=None,
                       help='视频级推理报告 JSON 文件 (timeline 模式)')
    parser.add_argument('--output', type=str, default='runs/visualize',
                       help='输出目录')
    
    args = parser.parse_args()
    
    if args.mode == 'deviation':
        if not args.image or not args.weights:
            print("deviation 模式需要 --image 和 --weights 参数")
            return
        visualize_deviation_map(args.image, args.weights, args.output)
    
    elif args.mode == 'curves':
        if not args.results_dir:
            print("curves 模式需要 --results_dir 参数")
            return
        plot_training_curves(args.results_dir, args.output)
    
    elif args.mode == 'ablation':
        if not args.results_file:
            args.results_file = 'runs/ablation/ablation_results.json'
        plot_ablation_bar_chart(args.results_file, args.output)
    
    elif args.mode == 'tvad':
        if not args.results_file:
            args.results_file = 'runs/ablation/ablation_results.json'
        plot_tvad_experiment(args.results_file, args.output)
    
    elif args.mode == 'timeline':
        if not args.video_report:
            args.video_report = 'runs/inference/video_report.json'
        plot_temporal_timeline(args.video_report, args.output)


if __name__ == "__main__":
    main()
