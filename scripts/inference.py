"""
推理与模型导出脚本

功能:
- 图片/视频/摄像头推理
- ONNX / TensorRT 模型导出
- 批量推理并保存结果
- 推理结果统计
- 视频级推理: 逐帧检测 + TVAD 三维度聚合判定
- 批量视频推理: 目录内所有视频逐一分析

使用方法:
    python scripts/inference.py --source test_images/ --weights best.pt
    python scripts/inference.py --source test.mp4 --weights best.pt
    python scripts/inference.py --source test.mp4 --video  # 视频级判定 (TVAD)
    python scripts/inference.py --source video_dir/ --video  # 批量视频判定
    python scripts/inference.py --export onnx --weights best.pt

    # TVAD 参数调节
    python scripts/inference.py --source test.mp4 --video --temporal_window 3.0 --suppression_alpha 0.4
"""
import os
import sys
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings('ignore', message='.*does not have a deterministic implementation.*')

INFERENCE_DIR = os.path.join(PROJECT_ROOT, 'runs', 'inference')

from models.register_modules import register_custom_modules
register_custom_modules()

from ultralytics import YOLO
from models.modules.tvad import (
    TVAD, VideoDecisionConfig, FrameDetection,
    create_tvad, CategoryThreshold,
)


CLASS_NAMES = {0: '抛洒物', 1: '机动车违停', 2: '逆行'}
CLASS_NAMES_EN = {0: 'debris', 1: 'illegal_parking', 2: 'retrograde'}

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv'}


def inference(args):
    """执行帧级推理"""
    os.chdir(PROJECT_ROOT)
    model = YOLO(args.weights)

    print(f"\n{'='*60}")
    print(f"  道路异常事件检测 - 帧级推理")
    print(f"  模型: {args.weights}")
    print(f"  输入: {args.source}")
    print(f"  置信度阈值: {args.conf}")
    print(f"{'='*60}")

    detection_stats = defaultdict(int)
    total_frames = 0

    results = model.predict(
        source=args.source, conf=args.conf, iou=args.iou,
        imgsz=args.imgsz, device=args.device, save=True,
        save_txt=args.save_txt, save_conf=args.save_txt,
        stream=True, project=INFERENCE_DIR,
        name='predict', exist_ok=True,
    )

    for r in results:
        total_frames += 1
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detection_stats[cls_id] += 1

    print(f"\n  推理完成! 处理帧数: {total_frames}")
    total_detections = sum(detection_stats.values())
    for cls_id in sorted(detection_stats.keys()):
        count = detection_stats[cls_id]
        name = CLASS_NAMES.get(cls_id, f'class_{cls_id}')
        print(f"    {name}: {count} ({count/max(total_detections,1)*100:.1f}%)")
    print(f"    总计: {total_detections}")


def _build_tvad(args) -> TVAD:
    """根据命令行参数构建 TVAD 实例"""
    config = VideoDecisionConfig(
        temporal_window=args.temporal_window,
        suppression_alpha=args.suppression_alpha,
    )
    return TVAD(config)


def video_inference(args):
    """
    视频级推理 —— 逐帧检测 + TVAD 三维度聚合判定

    流程:
    1. 逐帧推理获取所有检测框
    2. 收集 FrameDetection 列表
    3. 调用 TVAD.decide() 进行三维度评分
    4. 输出视频级判定结果 + JSON 报告
    """
    os.chdir(PROJECT_ROOT)
    model = YOLO(args.weights)
    tvad = _build_tvad(args)

    source = args.source

    # 收集视频文件列表
    if os.path.isdir(source):
        video_files = sorted([
            os.path.join(source, f) for f in os.listdir(source)
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS
        ])
    else:
        video_files = [source]

    if not video_files:
        print(f"[!] 未找到视频文件: {source}")
        return []

    print(f"\n{'='*60}")
    print(f"  道路异常事件检测 - 视频级推理 (TVAD)")
    print(f"  模型: {args.weights}")
    print(f"  视频数: {len(video_files)}")
    print(f"  TVAD 滑窗: {args.temporal_window}s")
    print(f"  孤立帧抑制: α={args.suppression_alpha}")
    print(f"{'='*60}")

    all_video_results = []

    for vi, vpath in enumerate(video_files, 1):
        vname = os.path.basename(vpath)
        print(f"\n  [{vi}/{len(video_files)}] 处理: {vname}")

        # 获取视频元信息
        import cv2
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 逐帧推理
        results = model.predict(
            source=vpath, conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=args.device, save=False,
            stream=True, verbose=False,
        )

        frame_detections = []
        frame_idx = 0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = [float(x) for x in box.xyxy[0]]

                frame_detections.append(FrameDetection(
                    frame_idx=frame_idx,
                    cls_id=cls_id,
                    confidence=conf,
                    bbox=bbox,
                ))
            frame_idx += 1

        actual_frames = frame_idx

        # TVAD 三维度聚合决策
        decision = tvad.decide(frame_detections, actual_frames, fps=fps)

        # 打印报告
        report_str = tvad.format_report(decision, vname)
        print(report_str)

        # 收集结果
        result_dict = tvad.to_dict(decision)
        result_dict['video'] = vname
        result_dict['video_path'] = vpath
        result_dict['fps'] = fps
        all_video_results.append(result_dict)

    # 保存 JSON 报告
    report_path = os.path.join(INFERENCE_DIR, 'video_report.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_video_results, f, ensure_ascii=False, indent=2)

    # 汇总
    print(f"\n{'='*60}")
    print(f"  视频级推理汇总")
    print(f"{'='*60}")
    print(f"  {'视频':<30} {'主事件':<15} {'分数':>10} {'帧占比':>8} "
          f"{'时序τ':>8} {'置信度':>8}")
    print("-" * 85)

    for res in all_video_results:
        pe = res.get('primary_event')
        if pe:
            print(f"  {res['video']:<30} {pe['name_cn']:<13} "
                  f"{pe['score']:>10.6f} {pe['frame_ratio']:>8.4f} "
                  f"{pe['temporal_consistency']:>8.4f} {pe['avg_confidence']:>8.4f}")
        else:
            print(f"  {res['video']:<30} {'无':^13}")

    print(f"\n  报告已保存: {report_path}")
    return all_video_results


def export_model(args):
    """导出模型"""
    os.chdir(PROJECT_ROOT)
    model = YOLO(args.weights)

    export_kwargs = {'format': args.export, 'imgsz': args.imgsz}

    if args.export == 'onnx':
        export_kwargs.update({'simplify': True, 'opset': 12, 'dynamic': False})
    elif args.export == 'engine':
        export_kwargs.update({'half': True, 'workspace': 4})

    exported_path = model.export(**export_kwargs)
    print(f"\n  ✅ 导出成功: {exported_path}")


def main():
    parser = argparse.ArgumentParser(description='推理与导出脚本')
    parser.add_argument('--weights', type=str,
                       default='runs/road_anomaly/yolov11m_improved/weights/best.pt')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--save_txt', action='store_true')

    # 视频级推理 (TVAD)
    parser.add_argument('--video', action='store_true',
                       help='启用视频级推理模式 (TVAD 三维度聚合判定)')
    parser.add_argument('--temporal_window', type=float, default=2.0,
                       help='TVAD 时序一致性滑窗大小 (秒), 默认 2.0')
    parser.add_argument('--suppression_alpha', type=float, default=0.3,
                       help='TVAD 孤立帧抑制因子 (0~1), 默认 0.3')

    # 导出
    parser.add_argument('--export', type=str, default=None,
                       choices=['onnx', 'engine', 'torchscript', 'openvino'])

    args = parser.parse_args()

    if args.export:
        export_model(args)
    elif args.source and args.video:
        video_inference(args)
    elif args.source:
        inference(args)
    else:
        print("请指定 --source (推理) 或 --export (导出)")
        parser.print_help()


if __name__ == "__main__":
    main()
