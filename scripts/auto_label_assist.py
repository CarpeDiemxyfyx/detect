"""
半自动标注辅助脚本
使用预训练模型 (YOLOv11) 对抽帧后的图片进行预标注,
然后人工修正, 大幅提升标注效率

工作流程:
1. 用通用目标检测模型对图片预推理
2. 将车辆检测结果映射为 illegal_parking / retrograde 候选
3. 将小物体检测结果映射为 debris 候选
4. 输出 YOLO 格式标注文件 (需人工审核修正)

使用方法:
    python scripts/auto_label_assist.py --source dataset/images/all --output dataset/labels/all
"""
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def auto_label(source_dir: str, output_dir: str, model_name: str = 'yolo11m.pt',
               conf: float = 0.3, device: str = '0'):
    """
    使用预训练模型辅助标注
    
    映射策略:
    - COCO 类别中的小物体/非常规物体 → 候选 debris (class 0)
    - COCO 类别中的 car/truck/bus → 候选 illegal_parking (class 1) 或 retrograde (class 2)
    - 需要根据文件名前缀 (debris_/illegal_parking_/retrograde_) 来判断主类别
    """
    from ultralytics import YOLO
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(model_name)
    
    # COCO 类别映射
    # 车辆类: 2=car, 5=bus, 7=truck
    vehicle_classes = {2, 5, 7}
    # 可能是抛洒物的小物体类: 24=backpack, 25=umbrella, 26=handbag, 28=suitcase, etc.
    debris_candidate_classes = {24, 25, 26, 28, 39, 56, 57, 58, 60, 62, 63, 67}
    
    image_files = sorted([
        f for f in Path(source_dir).iterdir()
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    ])
    
    print(f"\n{'='*60}")
    print(f"  半自动标注辅助")
    print(f"  图片数: {len(image_files)}")
    print(f"  模型: {model_name}")
    print(f"{'='*60}")
    
    labeled_count = 0
    
    for img_path in tqdm(image_files, desc="预标注"):
        # 根据文件名判断主类别
        fname = img_path.stem
        if fname.startswith('debris'):
            target_class = 0
        elif fname.startswith('illegal_parking'):
            target_class = 1
        elif fname.startswith('retrograde'):
            target_class = 2
        else:
            target_class = -1  # 未知, 按检测结果判断
        
        # 推理
        results = model(str(img_path), conf=conf, verbose=False, device=device)
        
        label_lines = []
        
        for r in results:
            img_h, img_w = r.orig_shape
            
            for box in r.boxes:
                coco_cls = int(box.cls[0])
                box_conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 转换为 YOLO 格式
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                
                # 映射类别
                if target_class == 0:
                    # 抛洒物帧: 所有小物体都标为 debris
                    if coco_cls in debris_candidate_classes or coco_cls not in vehicle_classes:
                        # 过滤掉太大的检测框 (可能是车辆)
                        if w * h < 0.15:  # 面积比小于15%
                            label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
                elif target_class == 1:
                    # 违停帧: 车辆标为 illegal_parking
                    if coco_cls in vehicle_classes:
                        label_lines.append(f"1 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
                elif target_class == 2:
                    # 逆行帧: 车辆标为 retrograde
                    if coco_cls in vehicle_classes:
                        label_lines.append(f"2 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                
                else:
                    # 未知类别: 尝试自动判断
                    if coco_cls in vehicle_classes:
                        label_lines.append(f"1 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                    elif w * h < 0.1:
                        label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        # 保存标签
        if label_lines:
            label_path = os.path.join(output_dir, fname + '.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines) + '\n')
            labeled_count += 1
    
    print(f"\n  ✅ 预标注完成!")
    print(f"  标注文件数: {labeled_count} / {len(image_files)}")
    print(f"  输出目录: {output_dir}")
    print(f"\n  ⚠️  请使用 LabelImg 对预标注结果进行人工审核修正!")
    print(f"  提示: 运行 labelimg {source_dir} {output_dir}/classes.txt")
    
    # 生成 classes.txt
    classes_file = os.path.join(output_dir, 'classes.txt')
    with open(classes_file, 'w') as f:
        f.write('debris\nillegal_parking\nretrograde\n')
    print(f"  类别文件: {classes_file}")


def main():
    parser = argparse.ArgumentParser(description='半自动标注辅助')
    parser.add_argument('--source', type=str, default='dataset/images/all')
    parser.add_argument('--output', type=str, default='dataset/labels/all')
    parser.add_argument('--model', type=str, default='yolo11m.pt')
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='0')
    
    args = parser.parse_args()
    auto_label(args.source, args.output, args.model, args.conf, args.device)


if __name__ == "__main__":
    main()
