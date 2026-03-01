"""
消融实验脚本
验证三个创新点各自以及组合的有效性

实验设计:
    A0: YOLOv11m Baseline (无改进)
    A1: YOLOv11m + SADR (仅创新点1)
    A2: YOLOv11m + BDFR (仅创新点2)
    A3: YOLOv11m + SADR + BDFR (双模块组合)
    A4: YOLOv11m + SADR + BDFR + TVAD (完整改进, 含视频级评估)

指标输出:
    - 帧级: mAP50, mAP50-95, Precision, Recall
    - 视频级 (A4): V-Acc, V-F1

使用方法:
    python scripts/ablation_study.py --epochs 200 --batch 16
    python scripts/ablation_study.py --experiments A0 A3 A4
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings('ignore', message='.*does not have a deterministic implementation.*')

ABLATION_DIR = os.path.join(PROJECT_ROOT, 'runs', 'ablation')

from models.register_modules import register_custom_modules
register_custom_modules()

from ultralytics import YOLO


# 消融实验配置
ABLATION_CONFIGS = {
    'A0': {
        'yaml': 'models/yolov11m-baseline.yaml',
        'name': 'A0_baseline',
        'desc': 'YOLOv11m Baseline',
        'sadr': False, 'bdfr': False, 'tvad': False,
    },
    'A1': {
        'yaml': 'models/yolov11m-sadr.yaml',
        'name': 'A1_sadr',
        'desc': 'YOLOv11m + SADR',
        'sadr': True, 'bdfr': False, 'tvad': False,
    },
    'A2': {
        'yaml': 'models/yolov11m-bdfr.yaml',
        'name': 'A2_bdfr',
        'desc': 'YOLOv11m + BDFR',
        'sadr': False, 'bdfr': True, 'tvad': False,
    },
    'A3': {
        'yaml': 'models/yolov11m-road-anomaly.yaml',
        'name': 'A3_full',
        'desc': 'YOLOv11m + SADR + BDFR',
        'sadr': True, 'bdfr': True, 'tvad': False,
    },
    'A4': {
        'yaml': 'models/yolov11m-road-anomaly.yaml',
        'name': 'A4_tvad',
        'desc': 'YOLOv11m + SADR + BDFR + TVAD',
        'sadr': True, 'bdfr': True, 'tvad': True,
    },
}


def count_parameters(model) -> float:
    """计算模型参数量 (M)"""
    try:
        total = sum(p.numel() for p in model.model.parameters())
        return total / 1e6
    except Exception:
        return 0.0


def run_ablation(args):
    """运行消融实验"""
    os.chdir(PROJECT_ROOT)
    
    # 确定要运行的实验
    if args.experiments:
        experiments = {k: v for k, v in ABLATION_CONFIGS.items() 
                      if k in args.experiments}
    else:
        experiments = ABLATION_CONFIGS
    
    results_all = {}
    
    print("\n" + "=" * 70)
    print("  消融实验 - 道路异常事件检测")
    print(f"  实验数: {len(experiments)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    for exp_id, config in experiments.items():
        print(f"\n{'='*60}")
        print(f"  实验 {exp_id}: {config['desc']}")
        print(f"  SADR: {'✓' if config['sadr'] else '✗'}")
        print(f"  BDFR: {'✓' if config['bdfr'] else '✗'}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 创建模型
        model = YOLO(config['yaml'])
        
        # 加载预训练权重
        pretrained = args.pretrained or 'yolo11m.pt'
        if os.path.exists(pretrained):
            model = model.load(pretrained)
        
        # 统计参数量
        params_m = count_parameters(model)
        
        # 训练
        model.train(
            data='dataset/road_anomaly.yaml',
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer='AdamW',
            lr0=0.0005,
            lrf=0.01,
            warmup_epochs=3,
            patience=args.patience,
            device=args.device,
            workers=args.workers,
            amp=True,
            cos_lr=True,
            close_mosaic=10,
            mosaic=1.0,
            mixup=0.15,
            project=ABLATION_DIR,
            name=config['name'],
            exist_ok=True,
            plots=True,
        )
        
        # 评估
        best_path = os.path.join(ABLATION_DIR, config['name'], 'weights', 'best.pt')
        if os.path.exists(best_path):
            best_model = YOLO(best_path)
            metrics = best_model.val(
                data='dataset/road_anomaly.yaml',
                imgsz=args.imgsz,
                device=args.device,
            )
            
            elapsed = time.time() - start_time
            
            results_all[exp_id] = {
                'name': config['desc'],
                'sadr': config['sadr'],
                'bdfr': config['bdfr'],
                'tvad': config.get('tvad', False),
                'mAP50': float(metrics.box.map50),
                'mAP50_95': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
                'params_M': round(params_m, 2),
                'training_time_min': round(elapsed / 60, 1),
            }
            
            # 如果有各类别AP
            try:
                per_class = {}
                class_names = ['debris', 'illegal_parking', 'retrograde']
                for i, name in enumerate(class_names):
                    per_class[name] = {
                        'AP50': float(metrics.box.ap50[i]),
                        'AP50_95': float(metrics.box.ap[i]),
                    }
                results_all[exp_id]['per_class'] = per_class
            except Exception:
                pass
            
            # 视频级评估 (仅 TVAD 实验)
            if config.get('tvad', False):
                try:
                    from scripts.evaluate import evaluate_video_level
                    v_metrics = evaluate_video_level(
                        best_path, 'dataset/road_anomaly.yaml',
                        test_video_dir='data', device=args.device,
                        imgsz=args.imgsz,
                    )
                    results_all[exp_id]['V_Acc'] = v_metrics.get('V_Acc', 0.0)
                    results_all[exp_id]['V_F1'] = v_metrics.get('V_F1', 0.0)
                    results_all[exp_id]['V_Precision'] = v_metrics.get('V_Precision', 0.0)
                    results_all[exp_id]['V_Recall'] = v_metrics.get('V_Recall', 0.0)
                except Exception as e:
                    print(f"  [!] 视频级评估失败: {e}")
                    results_all[exp_id]['V_Acc'] = 0.0
                    results_all[exp_id]['V_F1'] = 0.0
    
    # 保存结果
    os.makedirs(ABLATION_DIR, exist_ok=True)
    results_file = os.path.join(ABLATION_DIR, 'ablation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    
    # 打印汇总表格
    print("\n" + "=" * 105)
    print("  消融实验结果汇总")
    print("=" * 105)
    print(f"  {'实验':<6} {'模型':<30} {'SADR':^5} {'BDFR':^5} {'TVAD':^5} "
          f"{'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} {'V-Acc':>7} {'V-F1':>7} {'Params':>8}")
    print("-" * 105)
    
    for exp_id, res in results_all.items():
        sadr = '✓' if res['sadr'] else '✗'
        bdfr = '✓' if res['bdfr'] else '✗'
        tvad = '✓' if res.get('tvad', False) else '✗'
        v_acc = f"{res['V_Acc']:.4f}" if 'V_Acc' in res else '--'
        v_f1 = f"{res['V_F1']:.4f}" if 'V_F1' in res else '--'
        print(f"  {exp_id:<6} {res['name']:<30} {sadr:^5} {bdfr:^5} {tvad:^5} "
              f"{res['mAP50']:>8.4f} {res['mAP50_95']:>10.4f} "
              f"{res['precision']:>8.4f} {res['recall']:>8.4f} "
              f"{v_acc:>7} {v_f1:>7} "
              f"{res['params_M']:>7.2f}M")
    
    print("=" * 105)
    print(f"\n  结果已保存: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='消融实验脚本')
    parser.add_argument('--experiments', nargs='+', default=None,
                       choices=['A0', 'A1', 'A2', 'A3', 'A4'],
                       help='指定运行的实验 (默认全部)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pretrained', type=str, default=None)
    
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
