"""
数据增强工具
对标注好的数据集进行离线增强, 生成更多训练样本

增强策略:
- 随机裁剪 + 缩放
- 颜色抖动 (模拟不同光照/天气)
- 高斯噪声 (模拟低质量摄像头)
- 运动模糊 (模拟摄像头抖动)
- 随机遮挡 (提升鲁棒性)

使用方法:
    python scripts/augment_data.py --source dataset/images/train --labels dataset/labels/train --multiply 3
"""
import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random


def apply_color_jitter(image, brightness=0.3, contrast=0.3, saturation=0.3):
    """颜色抖动"""
    img = image.astype(np.float32)
    
    # 亮度
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-brightness, brightness)
        img = img * factor
    
    # 对比度
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = img.mean()
        img = (img - mean) * factor + mean
    
    # 饱和度 (HSV空间)
    if random.random() < 0.5:
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        factor = 1.0 + random.uniform(-saturation, saturation)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_gaussian_noise(image, sigma_range=(5, 25)):
    """添加高斯噪声"""
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_motion_blur(image, kernel_size_range=(5, 15)):
    """运动模糊"""
    k = random.choice(range(kernel_size_range[0], kernel_size_range[1], 2))
    angle = random.uniform(0, 360)
    
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = 1.0 / k
    
    # 旋转核
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel = kernel / kernel.sum()
    
    return cv2.filter2D(image, -1, kernel)


def apply_random_erasing(image, prob=0.3, area_ratio=(0.02, 0.15)):
    """随机遮挡 (Random Erasing)"""
    if random.random() > prob:
        return image
    
    h, w = image.shape[:2]
    area = h * w
    
    target_area = random.uniform(*area_ratio) * area
    aspect_ratio = random.uniform(0.3, 3.0)
    
    eh = int(np.sqrt(target_area * aspect_ratio))
    ew = int(np.sqrt(target_area / aspect_ratio))
    
    if eh < h and ew < w:
        y = random.randint(0, h - eh)
        x = random.randint(0, w - ew)
        image[y:y+eh, x:x+ew] = np.random.randint(0, 255, (eh, ew, 3), dtype=np.uint8)
    
    return image


def augment_single(image, augment_id):
    """对单张图片应用随机增强组合"""
    aug_img = image.copy()
    
    # 根据 augment_id 选择不同增强组合
    if augment_id % 3 == 0:
        aug_img = apply_color_jitter(aug_img)
        if random.random() < 0.3:
            aug_img = apply_gaussian_noise(aug_img)
    elif augment_id % 3 == 1:
        if random.random() < 0.5:
            aug_img = apply_motion_blur(aug_img)
        aug_img = apply_color_jitter(aug_img, brightness=0.4, contrast=0.2)
    else:
        aug_img = apply_color_jitter(aug_img, saturation=0.5)
        aug_img = apply_random_erasing(aug_img)
    
    # 随机翻转 (水平, 不影响标签因为标签x坐标会镜像)
    if random.random() < 0.5:
        aug_img = cv2.flip(aug_img, 1)
        return aug_img, True  # 需要翻转标签
    
    return aug_img, False


def flip_labels(label_content: str) -> str:
    """水平翻转标签"""
    flipped_lines = []
    for line in label_content.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = parts[0]
            x_center = 1.0 - float(parts[1])  # 水平翻转 x
            y_center = parts[2]
            w = parts[3]
            h = parts[4]
            flipped_lines.append(f"{cls_id} {x_center:.6f} {y_center} {w} {h}")
    return '\n'.join(flipped_lines)


def augment_dataset(source_dir: str, label_dir: str, multiply: int = 3):
    """
    对数据集进行离线增强
    原图保留, 生成 multiply 倍的增强样本
    """
    image_files = sorted([
        f for f in Path(source_dir).iterdir()
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    ])
    
    print(f"\n{'='*60}")
    print(f"  数据增强")
    print(f"  原始图片数: {len(image_files)}")
    print(f"  增强倍数: {multiply}")
    print(f"  预期增强后: {len(image_files) * (1 + multiply)}")
    print(f"{'='*60}")
    
    aug_count = 0
    
    for img_path in tqdm(image_files, desc="增强中"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # 读取对应标签
        label_path = Path(label_dir) / (img_path.stem + '.txt')
        label_content = ''
        if label_path.exists():
            with open(label_path, 'r') as f:
                label_content = f.read()
        
        for i in range(multiply):
            aug_img, flipped = augment_single(image, i)
            
            # 保存增强图片
            aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            cv2.imwrite(str(img_path.parent / aug_name), aug_img)
            
            # 保存对应标签
            if label_content:
                aug_label = label_content
                if flipped:
                    aug_label = flip_labels(label_content)
                
                aug_label_path = Path(label_dir) / f"{img_path.stem}_aug{i}.txt"
                with open(aug_label_path, 'w') as f:
                    f.write(aug_label)
            
            aug_count += 1
    
    print(f"\n  ✅ 增强完成! 生成 {aug_count} 张增强图片")


def main():
    parser = argparse.ArgumentParser(description='数据增强工具')
    parser.add_argument('--source', type=str, default='dataset/images/train')
    parser.add_argument('--labels', type=str, default='dataset/labels/train')
    parser.add_argument('--multiply', type=int, default=3,
                       help='增强倍数 (每张图生成N张增强样本)')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    augment_dataset(args.source, args.labels, args.multiply)


if __name__ == "__main__":
    main()
