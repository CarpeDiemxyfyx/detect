"""
数据集划分脚本
将提取的帧图片按 7:2:1 比例划分为 训练集/验证集/测试集

功能:
- 按类别分层采样, 确保各集合中类别比例一致
- 自动创建目录结构
- 支持已有标签文件的同步移动
- 生成数据统计报告

使用方法:
    python scripts/split_dataset.py --source dataset/images/all --labels dataset/labels/all
"""
import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


def get_category_from_filename(filename: str) -> str:
    """从文件名中提取类别名"""
    # 文件名格式: category_v01_0001.jpg
    # 提取 category 部分 (去掉 _v01_0001)
    parts = filename.split('_v')
    if len(parts) >= 2:
        return parts[0]
    # 兼容旧格式: category_000001.jpg
    parts = filename.rsplit('_', 1)
    if len(parts) >= 2:
        return parts[0]
    return 'unknown'


def get_video_group_from_filename(filename: str) -> str:
    """
    从文件名中提取视频分组标识 (category_v01)
    同一视频的所有帧会归为同一组，划分时不可拆分
    """
    # 文件名格式: category_v01_0001.jpg
    stem = Path(filename).stem
    parts = stem.rsplit('_', 1)  # ['category_v01', '0001']
    if len(parts) >= 2 and parts[0].count('_v') > 0:
        return parts[0]  # category_v01
    # 兼容旧格式: 按类别分组
    return get_category_from_filename(stem)


def split_dataset(source_dir: str, label_dir: str = None,
                  output_base: str = 'dataset', 
                  ratios: tuple = (0.7, 0.2, 0.1),
                  seed: int = 42):
    """
    划分数据集
    
    Args:
        source_dir: 源图片目录
        label_dir: 标签文件目录 (可选, 如果已标注)
        output_base: 输出根目录
        ratios: (训练, 验证, 测试) 比例
        seed: 随机种子
    """
    random.seed(seed)
    
    assert abs(sum(ratios) - 1.0) < 1e-6, "比例之和必须为1"
    
    # 收集所有图片, 按 (类别, 视频组) 二级分组
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # video_groups: { category: { video_group_key: [filename, ...] } }
    video_groups = defaultdict(lambda: defaultdict(list))
    
    source_path = Path(source_dir)
    for f in sorted(source_path.iterdir()):
        if f.suffix.lower() in image_extensions:
            cat = get_category_from_filename(f.stem)
            vg = get_video_group_from_filename(f.name)
            video_groups[cat][vg].append(f.name)
    
    if not video_groups:
        print(f"[!] 在 {source_dir} 中未找到图片文件")
        return
    
    # 创建目标目录
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_base, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_base, 'labels', split), exist_ok=True)
    
    # 按类别分层, 以视频组为最小单位进行划分 (防数据泄露)
    stats = {split: defaultdict(int) for split in splits}
    video_assignment = {}  # 记录每个视频组分到了哪个 split
    
    for category, groups in video_groups.items():
        group_keys = list(groups.keys())
        random.shuffle(group_keys)
        
        n_groups = len(group_keys)
        n_train = max(1, int(n_groups * ratios[0]))
        n_val = max(1, int(n_groups * ratios[1])) if n_groups > 2 else 0
        
        split_groups = {
            'train': group_keys[:n_train],
            'val': group_keys[n_train:n_train + n_val],
            'test': group_keys[n_train + n_val:]
        }
        
        for split, gkeys in split_groups.items():
            for gk in gkeys:
                video_assignment[gk] = split
                for fname in groups[gk]:
                    # 复制图片
                    src_img = os.path.join(source_dir, fname)
                    dst_img = os.path.join(output_base, 'images', split, fname)
                    shutil.copy2(src_img, dst_img)
                    
                    # 复制标签 (如果存在)
                    if label_dir:
                        label_name = Path(fname).stem + '.txt'
                        src_label = os.path.join(label_dir, label_name)
                        if os.path.exists(src_label):
                            dst_label = os.path.join(output_base, 'labels', split, label_name)
                            shutil.copy2(src_label, dst_label)
                    
                    stats[split][category] += 1
    
    # 打印统计报告
    print("\n" + "=" * 70)
    print("  数据集划分统计报告")
    print("=" * 70)
    print(f"  划分比例: 训练={ratios[0]:.0%}, 验证={ratios[1]:.0%}, 测试={ratios[2]:.0%}")
    print(f"  随机种子: {seed}")
    print("-" * 70)
    
    header = f"  {'类别':<20}" + "".join(f"{'  ' + s:<12}" for s in splits) + "  总计"
    print(header)
    print("-" * 70)
    
    all_categories = sorted(set().union(*[stats[s].keys() for s in splits]))
    
    for cat in all_categories:
        counts = [stats[s][cat] for s in splits]
        total = sum(counts)
        row = f"  {cat:<20}" + "".join(f"{c:<12}" for c in counts) + f"  {total}"
        print(row)
    
    # 总计行
    totals = [sum(stats[s].values()) for s in splits]
    total_all = sum(totals)
    print("-" * 70)
    print(f"  {'总计':<20}" + "".join(f"{t:<12}" for t in totals) + f"  {total_all}")
    print("=" * 70)
    
    # 打印视频组分配详情
    print(f"\n{'='*70}")
    print("  视频组分配详情 (同一视频的帧不跨split)")
    print("-" * 70)
    for split in splits:
        assigned = [k for k, v in video_assignment.items() if v == split]
        print(f"  {split}: {', '.join(sorted(assigned)) if assigned else '(无)'}")
    print("=" * 70)
    
    # 保存统计到文件
    stats_file = os.path.join(output_base, 'split_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"数据集划分统计\n")
        f.write(f"比例: train={ratios[0]}, val={ratios[1]}, test={ratios[2]}\n")
        f.write(f"随机种子: {seed}\n")
        f.write(f"划分单位: 视频组 (同一视频帧不跨split)\n\n")
        for split in splits:
            f.write(f"{split}:\n")
            for cat in all_categories:
                f.write(f"  {cat}: {stats[split][cat]}\n")
            f.write(f"  total: {sum(stats[split].values())}\n")
            assigned = [k for k, v in video_assignment.items() if v == split]
            f.write(f"  视频组: {', '.join(sorted(assigned))}\n\n")
    
    print(f"\n  统计信息已保存: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='数据集划分脚本')
    parser.add_argument('--source', type=str, default='dataset/images/all',
                       help='源图片目录')
    parser.add_argument('--labels', type=str, default='dataset/labels/all',
                       help='标签文件目录')
    parser.add_argument('--output', type=str, default='dataset',
                       help='输出根目录')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    
    label_dir = args.labels if os.path.exists(args.labels) else None
    
    split_dataset(
        source_dir=args.source,
        label_dir=label_dir,
        output_base=args.output,
        ratios=ratios,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
