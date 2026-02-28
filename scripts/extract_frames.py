"""
视频抽帧脚本
从三类视频数据中提取关键帧，并进行帧去重

功能:
- 按指定采样率从视频中抽帧
- 基于帧差法去除相似帧
- 支持图像质量筛选 (模糊检测)
- 统一命名规范
- 每视频最少帧保证: 当正常流程提取帧数不足 min_frames 时,
  从视频时间轴上均匀选取距离最远的帧进行回补

使用方法:
    python scripts/extract_frames.py --data_dir data --output_dir dataset/images/all
    python scripts/extract_frames.py --min_frames 5  # 每视频至少保留5帧
"""
import cv2
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


# 按类别自动采样率（可通过命令行覆盖）
# 策略：高密度采样 + 严格去重，保证有效帧充足且质量高
DEFAULT_CATEGORY_FPS = {
    'debris': 6,            # 抛洒物：小目标、形态多变，密采覆盖更多尺度
    'illegal_parking': 4,   # 违停：需前后帧对比判定，提高时序密度
    'retrograde': 8,        # 逆行：运动速度快，高密度防漏检
}


def compute_blur_score(image: np.ndarray) -> float:
    """
    计算图像模糊度 (拉普拉斯方差)
    值越小越模糊
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _save_frame(frame, output_dir: str, category: str,
                video_idx: int, frame_count: int) -> str:
    """保存单帧并返回文件路径"""
    fname = f"{category}_v{video_idx:02d}_{frame_count:04d}.jpg"
    save_path = os.path.join(output_dir, fname)
    cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return save_path


def _uniform_fallback(video_path: str, output_dir: str, category: str,
                      video_idx: int, existing_count: int,
                      min_frames: int, blur_threshold: float) -> dict:
    """
    时间均匀回补策略
    
    当正常流程提取的帧数 < min_frames 时调用。
    从视频时间轴上均匀选取 min_frames 个位置, 优先选择时间跨度最大的帧
    (首帧 → 尾帧 → 中间帧 → ...), 只做模糊检测, 不做帧差去重。
    已保存的帧对应的位置会被跳过。
    
    Returns:
        dict: {'补充帧数': int, 'skipped_blur': int}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'补充': 0, 'skipped_blur': 0}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return {'补充': 0, 'skipped_blur': 0}
    
    need = min_frames - existing_count
    if need <= 0:
        cap.release()
        return {'补充': 0, 'skipped_blur': 0}
    
    # 生成 候选位置: 均匀分布在时间轴上, 尽量保证首尾和中间都有
    # 多取一些候选, 以防部分被模糊过滤掉
    n_candidates = min(need * 3, total_frames)
    if n_candidates <= 1:
        candidate_indices = [0]
    else:
        step = max(1, (total_frames - 1) / (n_candidates - 1))
        candidate_indices = [int(round(i * step)) for i in range(n_candidates)]
    
    # 去重 + 排序, 保证时间跨度优先 (首、尾、中间交替)
    candidate_indices = sorted(set(candidate_indices))
    # 重排: 首帧 → 尾帧 → 中间帧 → 递归二分 (最大化时间间距)
    reordered = _maxdist_order(candidate_indices)
    
    补充 = 0
    skipped_blur = 0
    frame_seq = existing_count  # 续接编号
    
    for frame_pos in reordered:
        if 补充 >= need:
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 仅做模糊检测 (不做帧差去重, 因为目的是保证覆盖)
        blur_score = compute_blur_score(frame)
        if blur_score < blur_threshold * 0.5:  # 回补时放宽模糊阈值
            skipped_blur += 1
            continue
        
        _save_frame(frame, output_dir, category, video_idx, frame_seq)
        frame_seq += 1
        补充 += 1
    
    cap.release()
    return {'补充': 补充, 'skipped_blur': skipped_blur}


def _maxdist_order(indices: list) -> list:
    """
    将索引列表重排为最大时间间距优先顺序
    首帧 → 尾帧 → 中间帧 → 递归二分插入
    
    例: [0, 25, 50, 75, 100] → [0, 100, 50, 25, 75]
    """
    if len(indices) <= 2:
        return list(indices)
    
    result = []
    used = set()
    
    # 首尾
    result.append(indices[0])
    used.add(indices[0])
    result.append(indices[-1])
    used.add(indices[-1])
    
    # BFS 二分插入
    queue = [(0, len(indices) - 1)]  # (left_idx, right_idx) in indices list
    while queue and len(result) < len(indices):
        next_queue = []
        for li, ri in queue:
            mid = (li + ri) // 2
            if indices[mid] not in used:
                result.append(indices[mid])
                used.add(indices[mid])
            if mid - li > 1:
                next_queue.append((li, mid))
            if ri - mid > 1:
                next_queue.append((mid, ri))
        queue = next_queue
    
    # 补漏
    for idx in indices:
        if idx not in used:
            result.append(idx)
    
    return result


def extract_frames_from_video(video_path: str, output_dir: str, category: str,
                               video_idx: int, fps_sample: int = 2,
                               similarity_threshold: float = 15.0,
                               blur_threshold: float = 80.0,
                               max_frames_per_video: int = 200,
                               min_frames: int = 3) -> dict:
    """
    从单个视频中提取关键帧
    
    策略:
    1. 正常流程: 按 fps_sample 采样 → 帧差去重 → 模糊过滤
    2. 如果正常流程提取帧数 < min_frames, 启动时间均匀回补:
       从时间轴上均匀选取距离最远的帧, 放宽模糊阈值, 不做帧差去重
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        category: 类别英文名
        video_idx: 视频在该类别中的序号 (从1开始)
        fps_sample: 每秒采样帧数
        similarity_threshold: 帧差阈值 (低于则认为相似)
        blur_threshold: 模糊阈值 (低于则认为过于模糊)
        max_frames_per_video: 每个视频最多提取帧数
        min_frames: 每个视频最少保留帧数 (不足时触发时间均匀回补)
        
    Returns:
        dict: {'count': 提取帧数, 'skipped_similar': 跳过的相似帧数, 
               'skipped_blur': 跳过的模糊帧数, 'fallback': 回补帧数}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [!] 无法打开视频: {video_path}")
        return {'count': 0, 'skipped_similar': 0, 'skipped_blur': 0, 'fallback': 0}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    
    interval = max(1, int(fps / fps_sample))
    
    frame_count = 0
    skipped_similar = 0
    skipped_blur = 0
    prev_gray = None
    idx = 0
    
    while cap.isOpened() and frame_count < max_frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        
        if idx % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 帧差法去重
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                if np.mean(diff) < similarity_threshold:
                    skipped_similar += 1
                    idx += 1
                    continue
            
            # 模糊检测
            blur_score = compute_blur_score(gray)
            if blur_score < blur_threshold:
                skipped_blur += 1
                idx += 1
                continue
            
            # 保存帧 (命名: 类别_v视频序号_帧内序号)
            _save_frame(frame, output_dir, category, video_idx, frame_count)
            
            prev_gray = gray
            frame_count += 1
        
        idx += 1
    
    cap.release()
    
    # ====== 最少帧保证: 时间均匀回补 ======
    fallback_count = 0
    if frame_count < min_frames:
        fb = _uniform_fallback(
            video_path, output_dir, category, video_idx,
            existing_count=frame_count,
            min_frames=min_frames,
            blur_threshold=blur_threshold,
        )
        fallback_count = fb['补充']
        frame_count += fallback_count
        skipped_blur += fb['skipped_blur']
    
    return {
        'count': frame_count,
        'skipped_similar': skipped_similar,
        'skipped_blur': skipped_blur,
        'fallback': fallback_count,
    }


def extract_frames(video_dir: str, output_dir: str, category: str,
                   fps_sample: int = 2, similarity_threshold: float = 15.0,
                   blur_threshold: float = 80.0,
                   min_frames: int = 3) -> int:
    """
    从视频目录中提取所有视频的关键帧
    """
    os.makedirs(output_dir, exist_ok=True)
    
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv'}
    video_files = sorted([
        f for f in Path(video_dir).iterdir() 
        if f.suffix.lower() in video_extensions
    ])
    
    if not video_files:
        print(f"  [!] 目录 {video_dir} 中未找到视频文件")
        return 0
    
    print(f"\n{'='*60}")
    print(f"  类别: {category}")
    print(f"  视频目录: {video_dir}")
    print(f"  视频数量: {len(video_files)}")
    print(f"  采样率: {fps_sample} fps")
    print(f"  每视频最少帧: {min_frames}")
    print(f"{'='*60}")
    
    total_count = 0
    total_similar = 0
    total_blur = 0
    total_fallback = 0
    fallback_videos = 0
    
    for vid_idx, vid_path in enumerate(tqdm(video_files, desc=f"  [{category}] 抽帧"), start=1):
        result = extract_frames_from_video(
            video_path=str(vid_path),
            output_dir=output_dir,
            category=category,
            video_idx=vid_idx,
            fps_sample=fps_sample,
            similarity_threshold=similarity_threshold,
            blur_threshold=blur_threshold,
            min_frames=min_frames,
        )
        total_count += result['count']
        total_similar += result['skipped_similar']
        total_blur += result['skipped_blur']
        fb = result.get('fallback', 0)
        total_fallback += fb
        if fb > 0:
            fallback_videos += 1
    
    print(f"  [{category}] 结果: 提取 {total_count} 帧, "
          f"跳过相似帧 {total_similar}, 跳过模糊帧 {total_blur}")
    if total_fallback > 0:
        print(f"  [{category}] 回补: {fallback_videos} 个视频触发回补, "
              f"共补充 {total_fallback} 帧")
    
    return total_count


def main():
    parser = argparse.ArgumentParser(description='视频抽帧脚本')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='原始数据根目录')
    parser.add_argument('--output_dir', type=str, default='dataset/images/all',
                       help='输出图片目录')
    parser.add_argument('--fps', type=int, default=None,
                       help='全类别统一采样率；不填则按类别自动设置')
    parser.add_argument('--auto_fps', action=argparse.BooleanOptionalAction, default=True,
                       help='是否按类别自动设置采样率 (default: True)')
    parser.add_argument('--fps_debris', type=int, default=DEFAULT_CATEGORY_FPS['debris'],
                       help=f"抛洒物采样率 (default: {DEFAULT_CATEGORY_FPS['debris']})")
    parser.add_argument('--fps_illegal_parking', type=int, default=DEFAULT_CATEGORY_FPS['illegal_parking'],
                       help=f"机动车违停采样率 (default: {DEFAULT_CATEGORY_FPS['illegal_parking']})")
    parser.add_argument('--fps_retrograde', type=int, default=DEFAULT_CATEGORY_FPS['retrograde'],
                       help=f"逆行采样率 (default: {DEFAULT_CATEGORY_FPS['retrograde']})")
    parser.add_argument('--sim_thresh', type=float, default=15.0,
                       help='帧差相似度阈值，越低越严格 (default: 15.0)')
    parser.add_argument('--blur_thresh', type=float, default=80.0,
                       help='模糊度阈值，越高越严格 (default: 80.0)')
    parser.add_argument('--min_frames', type=int, default=3,
                       help='每视频最少保留帧数, 不足时均匀回补 (default: 3)')
    args = parser.parse_args()
    
    # 三类数据
    categories = {
        'debris': '抛洒物',
        'illegal_parking': '机动车违停',
        'retrograde': '逆行',
    }
    
    print("=" * 60)
    print("  道路异常事件检测 - 视频抽帧工具")
    print("=" * 60)

    if args.auto_fps and args.fps is None:
        category_fps_map = {
            'debris': args.fps_debris,
            'illegal_parking': args.fps_illegal_parking,
            'retrograde': args.fps_retrograde,
        }
        print("  采样策略: 按类别自动采样")
        print(f"    debris={category_fps_map['debris']} fps, "
              f"illegal_parking={category_fps_map['illegal_parking']} fps, "
              f"retrograde={category_fps_map['retrograde']} fps")
    else:
        unified_fps = args.fps if args.fps is not None else 2
        category_fps_map = {k: unified_fps for k in categories}
        print(f"  采样策略: 全类别统一采样 {unified_fps} fps")
    
    print(f"  每视频最少帧: {args.min_frames}")
    
    total = 0
    for eng_name, cn_name in categories.items():
        video_dir = os.path.join(args.data_dir, cn_name)
        if not os.path.exists(video_dir):
            print(f"\n  [!] 目录不存在: {video_dir}, 跳过")
            continue
        
        n = extract_frames(
            video_dir=video_dir,
            output_dir=args.output_dir,
            category=eng_name,
            fps_sample=category_fps_map[eng_name],
            similarity_threshold=args.sim_thresh,
            blur_threshold=args.blur_thresh,
            min_frames=args.min_frames,
        )
        total += n
    
    print(f"\n{'='*60}")
    print(f"  ✅ 抽帧完成! 总计提取 {total} 帧")
    print(f"  输出目录: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
