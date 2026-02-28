"""
对比实验脚本
与主流检测模型进行横向对比

对比模型:
    1. YOLOv5m
    2. YOLOv8m
    3. YOLOv11m (Baseline)
    4. RT-DETR-l
    5. Ours (改进YOLOv11m + TVAD)

指标:
    - 帧级: mAP50, mAP50-95, Precision, Recall
    - 视频级 (Ours): V-Acc, V-F1

使用方法:
    python scripts/comparison_experiment.py --epochs 200 --batch 16
    python scripts/comparison_experiment.py --models yolov8m ours
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.register_modules import register_custom_modules
register_custom_modules()

from ultralytics import YOLO


# 对比模型配置
COMPARISON_MODELS = {
    'yolov5m': {
        'weight': 'yolov5m.pt',
        'is_custom': False,
        'desc': 'YOLOv5m',
    },
    'yolov8m': {
        'weight': 'yolov8m.pt',
        'is_custom': False,
        'desc': 'YOLOv8m',
    },
    'yolov11m': {
        'weight': 'yolo11m.pt',
        'is_custom': False,
        'desc': 'YOLOv11m',
    },
    'rtdetr-l': {
        'weight': 'rtdetr-l.pt',
        'is_custom': False,
        'desc': 'RT-DETR-l',
    },
    'ours': {
        'yaml': 'models/yolov11-road-anomaly.yaml',
        'weight': 'yolo11m.pt',
        'is_custom': True,
        'desc': 'Ours (Improved YOLOv11m)',
        'tvad': True,
    },
}


def run_comparison(args):
    """运行对比实验"""
    os.chdir(PROJECT_ROOT)
    
    if args.models:
        models = {k: v for k, v in COMPARISON_MODELS.items() if k in args.models}
    else:
        models = COMPARISON_MODELS
    
    results_all = {}
    
    print("\n" + "=" * 70)
    print("  对比实验 - 道路异常事件检测")
    print(f"  对比模型数: {len(models)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    for model_name, config in models.items():
        print(f"\n{'='*60}")
        print(f"  模型: {config['desc']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if config['is_custom']:
                # 自定义改进模型
                model = YOLO(config['yaml'])
                if os.path.exists(config['weight']):
                    model = model.load(config['weight'])
            else:
                # 标准预训练模型
                model = YOLO(config['weight'])
            
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
                project='runs/comparison',
                name=model_name,
                exist_ok=True,
                plots=True,
            )
            
            # 评估
            best_path = f"runs/comparison/{model_name}/weights/best.pt"
            if os.path.exists(best_path):
                best_model = YOLO(best_path)
                metrics = best_model.val(
                    data='dataset/road_anomaly.yaml',
                    imgsz=args.imgsz,
                    device=args.device,
                )
                
                elapsed = time.time() - start_time
                
                # 计算参数量和GFLOPs
                try:
                    params = sum(p.numel() for p in best_model.model.parameters()) / 1e6
                except Exception:
                    params = 0.0
                
                results_all[model_name] = {
                    'name': config['desc'],
                    'mAP50': float(metrics.box.map50),
                    'mAP50_95': float(metrics.box.map),
                    'precision': float(metrics.box.mp),
                    'recall': float(metrics.box.mr),
                    'params_M': round(params, 2),
                    'training_time_min': round(elapsed / 60, 1),
                }
                
                # 视频级评估 (仅 Ours 模型)
                if config.get('tvad', False):
                    try:
                        from scripts.evaluate import evaluate_video_level
                        v_metrics = evaluate_video_level(
                            best_path, 'dataset/road_anomaly.yaml',
                            test_video_dir='data', device=args.device,
                            imgsz=args.imgsz,
                        )
                        results_all[model_name]['V_Acc'] = v_metrics.get('V_Acc', 0.0)
                        results_all[model_name]['V_F1'] = v_metrics.get('V_F1', 0.0)
                    except Exception as e:
                        print(f"  [!] 视频级评估失败: {e}")
                
        except Exception as e:
            print(f"  [!] 模型 {model_name} 训练失败: {e}")
            results_all[model_name] = {
                'name': config['desc'],
                'error': str(e),
            }
    
    # 保存结果
    os.makedirs('runs/comparison', exist_ok=True)
    results_file = 'runs/comparison/comparison_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    
    # 打印汇总
    print("\n" + "=" * 100)
    print("  对比实验结果汇总")
    print("=" * 100)
    print(f"  {'模型':<30} {'mAP50':>8} {'mAP50-95':>10} "
          f"{'P':>8} {'R':>8} {'V-Acc':>7} {'V-F1':>7} {'Params':>8}")
    print("-" * 100)
    
    for name, res in results_all.items():
        if 'error' in res:
            print(f"  {res['name']:<30} {'ERROR':>8}")
        else:
            v_acc = f"{res['V_Acc']:.4f}" if 'V_Acc' in res else '--'
            v_f1 = f"{res['V_F1']:.4f}" if 'V_F1' in res else '--'
            print(f"  {res['name']:<30} {res['mAP50']:>8.4f} "
                  f"{res['mAP50_95']:>10.4f} {res['precision']:>8.4f} "
                  f"{res['recall']:>8.4f} {v_acc:>7} {v_f1:>7} "
                  f"{res['params_M']:>7.2f}M")
    
    print("=" * 100)
    print(f"\n  结果已保存: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='对比实验脚本')
    parser.add_argument('--models', nargs='+', default=None,
                       help='指定运行的模型')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=4)
    
    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
