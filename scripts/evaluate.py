"""
模型评估脚本
对训练好的模型进行全面评估和可视化

功能:
- 计算 mAP, Precision, Recall 等指标
- 各类别详细精度分析
- 生成混淆矩阵
- 速度基准测试 (FPS)
- 模型复杂度分析 (参数量, GFLOPs)
- 偏离度图可视化 (BDFR模块)
- 视频级评估 (V-Acc, V-Precision, V-Recall, V-F1)

使用方法:
    python scripts/evaluate.py --weights runs/road_anomaly/yolov11m_improved/weights/best.pt
    python scripts/evaluate.py --weights best.pt --speed_test --visualize
    python scripts/evaluate.py --weights best.pt --video_eval --test_video_dir data
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.register_modules import register_custom_modules
register_custom_modules()

from ultralytics import YOLO


def evaluate_model(args):
    """全面评估模型"""
    os.chdir(PROJECT_ROOT)
    
    print(f"\n{'='*60}")
    print(f"  模型评估")
    print(f"  权重: {args.weights}")
    print(f"{'='*60}")
    
    model = YOLO(args.weights)
    
    # 1. 标准评估
    print("\n[1/4] 标准指标评估...")
    metrics = model.val(
        data='dataset/road_anomaly.yaml',
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
        save_json=True,
        conf=0.001,
        iou=0.6,
    )
    
    results = {
        'overall': {
            'mAP50': float(metrics.box.map50),
            'mAP50_95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }
    }
    
    # 各类别指标
    class_names = ['debris', 'illegal_parking', 'retrograde']
    cn_names = ['抛洒物', '机动车违停', '逆行']
    
    try:
        results['per_class'] = {}
        for i, (en, cn) in enumerate(zip(class_names, cn_names)):
            results['per_class'][en] = {
                'cn_name': cn,
                'AP50': float(metrics.box.ap50[i]),
                'AP50_95': float(metrics.box.ap[i]),
            }
    except Exception as e:
        print(f"  [!] 各类别指标提取失败: {e}")
    
    # 2. 速度测试
    if args.speed_test:
        print("\n[2/4] 速度基准测试...")
        import torch
        
        dummy = torch.randn(1, 3, args.imgsz, args.imgsz).to(args.device)
        
        # 预热
        for _ in range(50):
            model.predict(
                source=np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8),
                verbose=False, device=args.device
            )
        
        # 计时
        times = []
        for _ in range(200):
            img = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            model.predict(source=img, verbose=False, device=args.device)
            times.append(time.perf_counter() - t0)
        
        avg_time = np.mean(times) * 1000  # ms
        fps = 1000.0 / avg_time
        
        results['speed'] = {
            'avg_inference_ms': round(avg_time, 2),
            'fps': round(fps, 1),
        }
        print(f"  平均推理时间: {avg_time:.2f} ms")
        print(f"  FPS: {fps:.1f}")
    
    # 3. 模型复杂度
    print("\n[3/4] 模型复杂度分析...")
    try:
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        results['complexity'] = {
            'total_params': total_params,
            'total_params_M': round(total_params / 1e6, 2),
            'trainable_params': trainable_params,
            'trainable_params_M': round(trainable_params / 1e6, 2),
        }
        print(f"  总参数量: {total_params/1e6:.2f}M")
        print(f"  可训练参数: {trainable_params/1e6:.2f}M")
    except Exception as e:
        print(f"  [!] 复杂度分析失败: {e}")
    
    # 4. 可视化测试
    if args.visualize and os.path.exists('dataset/images/test'):
        print("\n[4/4] 可视化检测结果...")
        test_images = list(Path('dataset/images/test').glob('*.jpg'))[:20]
        if test_images:
            model.predict(
                source=[str(p) for p in test_images],
                conf=0.5,
                iou=0.45,
                save=True,
                device=args.device,
                project='runs/evaluate',
                name='visualize',
                exist_ok=True,
            )
            print(f"  可视化结果已保存到: runs/evaluate/visualize/")
    
    # 保存评估结果
    output_dir = os.path.dirname(args.weights) if os.path.dirname(args.weights) else 'runs/evaluate'
    os.makedirs(output_dir, exist_ok=True)
    
    eval_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("  评估结果汇总")
    print(f"{'='*60}")
    print(f"  Overall mAP@0.5:      {results['overall']['mAP50']:.4f}")
    print(f"  Overall mAP@0.5:0.95: {results['overall']['mAP50_95']:.4f}")
    print(f"  Precision:             {results['overall']['precision']:.4f}")
    print(f"  Recall:                {results['overall']['recall']:.4f}")
    
    if 'per_class' in results:
        print(f"\n  各类别 AP@0.5:")
        for en, data in results['per_class'].items():
            print(f"    {data['cn_name']} ({en}): {data['AP50']:.4f}")
    
    if 'speed' in results:
        print(f"\n  推理速度: {results['speed']['fps']:.1f} FPS "
              f"({results['speed']['avg_inference_ms']:.2f} ms/frame)")
    
    if 'complexity' in results:
        print(f"  模型参数量: {results['complexity']['total_params_M']:.2f}M")
    
    print(f"\n  评估结果已保存: {eval_file}")
    print(f"{'='*60}")


# ============================================================
# 视频级评估 (V-Acc, V-Precision, V-Recall, V-F1)
# ============================================================

def evaluate_video_level(
    weights_path: str,
    data_yaml: str = 'dataset/road_anomaly.yaml',
    test_video_dir: str = 'data',
    device: str = '0',
    imgsz: int = 640,
    conf: float = 0.5,
    iou: float = 0.45,
    temporal_window: float = 2.0,
    suppression_alpha: float = 0.3,
) -> dict:
    """
    视频级评估: 对测试视频进行 TVAD 聚合判定, 与 ground-truth 对比
    
    Ground-truth 规则:
        视频所在目录名决定其标签:
        - data/抛洒物/ → debris (cls 0)
        - data/机动车违停/ → illegal_parking (cls 1)
        - data/逆行/ → retrograde (cls 2)
    
    指标:
        V-Acc: 视频级准确率 (正确判定 / 总视频数)
        V-Precision: 各类别加权精确率
        V-Recall: 各类别加权召回率
        V-F1: 各类别加权 F1-score
    
    Args:
        weights_path: 模型权重路径
        data_yaml: 数据集配置文件
        test_video_dir: 测试视频根目录 (包含类别子目录)
        device: 推理设备
        imgsz: 推理图片尺寸
        conf: 置信度阈值
        iou: IOU 阈值
        temporal_window: TVAD 滑窗大小
        suppression_alpha: TVAD 孤立帧抑制因子
    
    Returns:
        dict: {'V_Acc': ..., 'V_Precision': ..., 'V_Recall': ..., 'V_F1': ...,
               'confusion': {...}, 'per_video': [...]}
    """
    from models.modules.tvad import TVAD, VideoDecisionConfig, FrameDetection
    import cv2
    
    # 类别目录映射
    CATEGORY_MAP = {
        '抛洒物': 0,
        '机动车违停': 1,
        '逆行': 2,
    }
    
    video_ext = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv'}
    
    # 收集测试视频及其 ground-truth
    test_videos = []  # [(video_path, gt_cls_id), ...]
    
    for cn_name, cls_id in CATEGORY_MAP.items():
        cat_dir = os.path.join(test_video_dir, cn_name)
        if not os.path.isdir(cat_dir):
            continue
        for f in sorted(os.listdir(cat_dir)):
            if Path(f).suffix.lower() in video_ext:
                test_videos.append((os.path.join(cat_dir, f), cls_id))
    
    if not test_videos:
        print(f"  [!] 在 {test_video_dir} 中未找到测试视频")
        return {'V_Acc': 0.0, 'V_Precision': 0.0, 'V_Recall': 0.0, 'V_F1': 0.0}
    
    print(f"\n  视频级评估: 共 {len(test_videos)} 个视频")
    
    # 加载模型和 TVAD
    model = YOLO(weights_path)
    tvad = TVAD(VideoDecisionConfig(
        temporal_window=temporal_window,
        suppression_alpha=suppression_alpha,
    ))
    
    # 逐视频推理 + TVAD 判定
    correct = 0
    per_video_results = []
    
    # 混淆矩阵: confusion[gt][pred] = count
    n_classes = len(CATEGORY_MAP)
    confusion = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    # 最后一行/列 = "无" (未判定出事件)
    
    for vpath, gt_cls in test_videos:
        vname = os.path.basename(vpath)
        
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        
        # 逐帧推理
        results = model.predict(
            source=vpath, conf=conf, iou=iou,
            imgsz=imgsz, device=device, save=False,
            stream=True, verbose=False,
        )
        
        frame_dets = []
        frame_idx = 0
        for r in results:
            for box in r.boxes:
                frame_dets.append(FrameDetection(
                    frame_idx=frame_idx,
                    cls_id=int(box.cls[0]),
                    confidence=float(box.conf[0]),
                ))
            frame_idx += 1
        
        # TVAD 判定
        decision = tvad.decide(frame_dets, frame_idx, fps=fps)
        
        pred_cls = decision.primary_event.cls_id if decision.primary_event else -1
        is_correct = (pred_cls == gt_cls)
        if is_correct:
            correct += 1
        
        # 更新混淆矩阵
        gt_idx = gt_cls
        pred_idx = pred_cls if pred_cls >= 0 else n_classes
        confusion[gt_idx][pred_idx] += 1
        
        per_video_results.append({
            'video': vname,
            'gt_cls': gt_cls,
            'pred_cls': pred_cls,
            'correct': is_correct,
            'score': decision.primary_event.score if decision.primary_event else 0.0,
        })
    
    # 计算指标
    total_videos = len(test_videos)
    v_acc = correct / max(total_videos, 1)
    
    # 各类别 Precision / Recall / F1
    precisions = []
    recalls = []
    f1s = []
    class_weights = []
    
    for cls_id in range(n_classes):
        tp = confusion[cls_id][cls_id]
        fp = confusion[:, cls_id].sum() - tp
        fn = confusion[cls_id, :].sum() - tp
        
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        
        n_gt = confusion[cls_id, :].sum()
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        class_weights.append(n_gt)
    
    total_gt = sum(class_weights)
    if total_gt > 0:
        v_precision = sum(p * w for p, w in zip(precisions, class_weights)) / total_gt
        v_recall = sum(r * w for r, w in zip(recalls, class_weights)) / total_gt
        v_f1 = sum(f * w for f, w in zip(f1s, class_weights)) / total_gt
    else:
        v_precision = v_recall = v_f1 = 0.0
    
    # 打印结果
    class_names_cn = ['抛洒物', '机动车违停', '逆行']
    print(f"\n  视频级评估结果:")
    print(f"    V-Acc:       {v_acc:.4f} ({correct}/{total_videos})")
    print(f"    V-Precision: {v_precision:.4f}")
    print(f"    V-Recall:    {v_recall:.4f}")
    print(f"    V-F1:        {v_f1:.4f}")
    print(f"\n    各类别:")
    for i, cn in enumerate(class_names_cn):
        print(f"      {cn}: P={precisions[i]:.4f} R={recalls[i]:.4f} F1={f1s[i]:.4f}")
    
    return {
        'V_Acc': round(v_acc, 4),
        'V_Precision': round(v_precision, 4),
        'V_Recall': round(v_recall, 4),
        'V_F1': round(v_f1, 4),
        'per_class': {
            class_names_cn[i]: {'P': precisions[i], 'R': recalls[i], 'F1': f1s[i]}
            for i in range(n_classes)
        },
        'confusion_matrix': confusion.tolist(),
        'per_video': per_video_results,
    }


def main():
    parser = argparse.ArgumentParser(description='模型评估脚本')
    parser.add_argument('--weights', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--speed_test', action='store_true',
                       help='是否进行速度测试')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化结果')
    parser.add_argument('--video_eval', action='store_true',
                       help='是否进行视频级评估 (TVAD)')
    parser.add_argument('--test_video_dir', type=str, default='data',
                       help='测试视频根目录 (包含类别子目录)')
    parser.add_argument('--temporal_window', type=float, default=2.0,
                       help='TVAD 滑窗大小 (秒)')
    parser.add_argument('--suppression_alpha', type=float, default=0.3,
                       help='TVAD 孤立帧抑制因子')
    
    args = parser.parse_args()
    
    # 帧级评估
    evaluate_model(args)
    
    # 视频级评估
    if args.video_eval:
        v_results = evaluate_video_level(
            args.weights,
            test_video_dir=args.test_video_dir,
            device=args.device,
            imgsz=args.imgsz,
            temporal_window=args.temporal_window,
            suppression_alpha=args.suppression_alpha,
        )
        
        # 保存视频级评估结果
        output_dir = os.path.dirname(args.weights) if os.path.dirname(args.weights) else 'runs/evaluate'
        os.makedirs(output_dir, exist_ok=True)
        v_eval_file = os.path.join(output_dir, 'video_evaluation_results.json')
        with open(v_eval_file, 'w', encoding='utf-8') as f:
            json.dump(v_results, f, indent=2, ensure_ascii=False)
        print(f"\n  视频级评估结果已保存: {v_eval_file}")


if __name__ == "__main__":
    main()
