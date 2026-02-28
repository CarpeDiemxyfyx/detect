"""
TVAD: Temporal-aware Video Aggregation Decision Module (创新点3)
时序感知视频聚合决策模块

核心创新:
1. 三维度评分公式 —— S_c = r_c × τ_c × conf_c
   - r_c: 帧占比 (检出帧数 / 总帧数)
   - τ_c: 时序一致性 (滑窗连续检出比率)
   - conf_c: 平均置信度
2. 滑窗时序一致性分析 —— 区分持续性事件与偶发误检
3. 各类别自适应阈值 —— 不同事件类型使用不同判定标准
4. 误报抑制机制 —— 孤立帧检出自动降权

适用场景:
    视频级事件判定 —— 将逐帧检测结果聚合为"该视频属于哪种异常事件"的决策
    本模块为纯推理阶段模块, 不参与训练

使用方法:
    from models.modules.tvad import TVAD, VideoDecisionConfig
    
    tvad = TVAD()
    result = tvad.decide(frame_detections, total_frames, fps=25.0)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math


# ============ 类别定义 ============
CLASS_NAMES = {0: '抛洒物', 1: '机动车违停', 2: '逆行'}
CLASS_NAMES_EN = {0: 'debris', 1: 'illegal_parking', 2: 'retrograde'}


@dataclass
class CategoryThreshold:
    """
    单类别判定阈值配置
    
    Attributes:
        min_score: 三维度综合分数最低阈值
        min_frame_ratio: 最低帧占比 r_c
        min_temporal_consistency: 最低时序一致性 τ_c
        min_avg_confidence: 最低平均置信度 conf_c
        min_det_frames: 最少检出帧数 (绝对值下限)
    """
    min_score: float = 0.005
    min_frame_ratio: float = 0.05
    min_temporal_consistency: float = 0.15
    min_avg_confidence: float = 0.30
    min_det_frames: int = 3


@dataclass
class VideoDecisionConfig:
    """
    TVAD 全局配置
    
    Attributes:
        temporal_window: 滑窗大小 (秒), 用于计算时序一致性
        fps: 默认帧率 (用于秒级换算)
        category_thresholds: 各类别自适应阈值
        suppression_alpha: 孤立帧抑制因子 (0~1, 越大抑制越强)
        score_eps: 数值稳定项
    """
    temporal_window: float = 2.0
    fps: float = 25.0
    category_thresholds: Dict[int, CategoryThreshold] = field(default_factory=lambda: {
        0: CategoryThreshold(  # 抛洒物: 静态目标, 帧占比通常较高
            min_score=0.003,
            min_frame_ratio=0.05,
            min_temporal_consistency=0.10,
            min_avg_confidence=0.30,
            min_det_frames=3,
        ),
        1: CategoryThreshold(  # 违停: 持续性静态事件, 时序一致性高
            min_score=0.003,
            min_frame_ratio=0.05,
            min_temporal_consistency=0.10,
            min_avg_confidence=0.30,
            min_det_frames=2,
        ),
        2: CategoryThreshold(  # 逆行: 短暂动态事件, 帧占比可能较低
            min_score=0.002,
            min_frame_ratio=0.03,
            min_temporal_consistency=0.08,
            min_avg_confidence=0.35,
            min_det_frames=3,
        ),
    })
    suppression_alpha: float = 0.3
    score_eps: float = 1e-8


@dataclass
class FrameDetection:
    """
    单帧检测结果
    
    Attributes:
        frame_idx: 帧序号 (0-based)
        cls_id: 类别 ID
        confidence: 置信度
        bbox: 边界框 [x1, y1, x2, y2] (可选, 用于空间一致性分析)
    """
    frame_idx: int
    cls_id: int
    confidence: float
    bbox: Optional[List[float]] = None


@dataclass
class VideoEventResult:
    """
    视频级事件判定结果 (单类别)
    
    Attributes:
        cls_id: 类别 ID
        name_cn: 类别中文名
        name_en: 类别英文名
        score: 三维度综合分数 S_c = r_c × τ_c × conf_c
        frame_ratio: 帧占比 r_c
        temporal_consistency: 时序一致性 τ_c
        avg_confidence: 平均置信度 conf_c
        det_frames: 检出帧数
        total_frames: 总帧数
        passed: 是否通过阈值判定
        suppression_applied: 是否触发了孤立帧抑制
    """
    cls_id: int
    name_cn: str
    name_en: str
    score: float
    frame_ratio: float
    temporal_consistency: float
    avg_confidence: float
    det_frames: int
    total_frames: int
    passed: bool
    suppression_applied: bool = False


@dataclass
class VideoDecisionResult:
    """
    视频聚合决策最终结果
    
    Attributes:
        primary_event: 主事件 (得分最高且通过阈值的事件)
        all_events: 所有类别的评估结果 (按 score 降序)
        total_frames: 视频总帧数
        duration_sec: 视频时长 (秒)
        timeline: 时间线统计 {秒: {cls_id: 检出数}}
    """
    primary_event: Optional[VideoEventResult]
    all_events: List[VideoEventResult]
    total_frames: int
    duration_sec: float
    timeline: Dict[int, Dict[int, int]]


class TVAD:
    """
    Temporal-aware Video Aggregation Decision Module
    时序感知视频聚合决策模块
    
    将逐帧检测结果聚合为视频级事件判定
    
    三维度评分公式:
        S_c = r_c × τ_c × conf_c
    
    其中:
        r_c = |{frame has cls c}| / total_frames          (帧占比)
        τ_c = mean(window_hit_ratio for each window)      (时序一致性)
        conf_c = mean(confidences of cls c)                (平均置信度)
    
    判定逻辑:
        1. 计算各类别三维度分数
        2. 对孤立帧检出进行抑制 (降低 τ_c)
        3. 按类别自适应阈值过滤
        4. 取最高分作为主事件
    
    Args:
        config: VideoDecisionConfig 配置对象
    """
    
    def __init__(self, config: Optional[VideoDecisionConfig] = None):
        self.config = config or VideoDecisionConfig()
    
    def compute_temporal_consistency(
        self,
        det_frame_indices: List[int],
        total_frames: int,
        fps: float
    ) -> Tuple[float, bool]:
        """
        计算时序一致性 τ_c
        
        使用滑动窗口方法:
        1. 将视频按 temporal_window 秒分割为多个窗口
        2. 每个窗口内计算检出帧占该窗口总帧数的比率
        3. τ_c = 所有窗口命中率的均值
        
        同时检测是否为孤立帧检出 (连续未检出段过长)
        
        Args:
            det_frame_indices: 检出帧的索引列表
            total_frames: 视频总帧数
            fps: 视频帧率
        
        Returns:
            (temporal_consistency, is_isolated)
            - temporal_consistency: τ_c 值 [0, 1]
            - is_isolated: 是否判定为孤立帧 (需要抑制)
        """
        if not det_frame_indices or total_frames <= 0:
            return 0.0, True
        
        det_set = set(det_frame_indices)
        window_size = max(1, int(fps * self.config.temporal_window))
        
        # 滑窗扫描
        n_windows = max(1, math.ceil(total_frames / window_size))
        window_hit_ratios = []
        windows_with_hits = 0
        
        for w in range(n_windows):
            w_start = w * window_size
            w_end = min((w + 1) * window_size, total_frames)
            w_len = w_end - w_start
            if w_len <= 0:
                continue
            
            hits = sum(1 for fi in range(w_start, w_end) if fi in det_set)
            ratio = hits / w_len
            window_hit_ratios.append(ratio)
            
            if hits > 0:
                windows_with_hits += 1
        
        if not window_hit_ratios:
            return 0.0, True
        
        tau_c = sum(window_hit_ratios) / len(window_hit_ratios)
        
        # 孤立帧检测: 如果命中窗口占比过低, 视为孤立帧
        window_coverage = windows_with_hits / len(window_hit_ratios)
        is_isolated = (window_coverage < 0.15 and len(det_set) < 5)
        
        return tau_c, is_isolated
    
    def compute_frame_ratio(
        self,
        det_frame_indices: List[int],
        total_frames: int
    ) -> float:
        """
        计算帧占比 r_c
        
        r_c = |unique_det_frames| / total_frames
        
        Args:
            det_frame_indices: 检出帧的索引列表
            total_frames: 视频总帧数
        
        Returns:
            r_c: 帧占比 [0, 1]
        """
        if total_frames <= 0:
            return 0.0
        unique_frames = len(set(det_frame_indices))
        return unique_frames / total_frames
    
    def compute_avg_confidence(self, confidences: List[float]) -> float:
        """
        计算平均置信度 conf_c
        
        Args:
            confidences: 该类别所有检出框的置信度列表
        
        Returns:
            conf_c: 平均置信度
        """
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)
    
    def decide(
        self,
        frame_detections: List[FrameDetection],
        total_frames: int,
        fps: Optional[float] = None,
    ) -> VideoDecisionResult:
        """
        视频级聚合决策 (核心方法)
        
        流程:
        1. 按类别归组检测结果
        2. 逐类别计算三维度分数
        3. 应用孤立帧抑制
        4. 按类别自适应阈值过滤
        5. 选出主事件
        
        Args:
            frame_detections: 所有帧的检测结果列表
            total_frames: 视频总帧数
            fps: 视频帧率 (None 时使用配置默认值)
        
        Returns:
            VideoDecisionResult: 视频级判定结果
        """
        fps = fps or self.config.fps
        duration_sec = total_frames / max(fps, 1.0)
        
        # 1. 按类别归组
        cls_frame_indices: Dict[int, List[int]] = defaultdict(list)
        cls_confidences: Dict[int, List[float]] = defaultdict(list)
        timeline: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        for det in frame_detections:
            cls_frame_indices[det.cls_id].append(det.frame_idx)
            cls_confidences[det.cls_id].append(det.confidence)
            sec = int(det.frame_idx / fps)
            timeline[sec][det.cls_id] += 1
        
        # 2. 逐类别计算三维度分数
        all_events: List[VideoEventResult] = []
        
        for cls_id in sorted(set(cls_frame_indices.keys()) | 
                             set(self.config.category_thresholds.keys())):
            det_indices = cls_frame_indices.get(cls_id, [])
            confs = cls_confidences.get(cls_id, [])
            
            if not det_indices:
                continue
            
            # 三维度计算
            r_c = self.compute_frame_ratio(det_indices, total_frames)
            tau_c, is_isolated = self.compute_temporal_consistency(
                det_indices, total_frames, fps
            )
            conf_c = self.compute_avg_confidence(confs)
            
            # 3. 孤立帧抑制
            suppression_applied = False
            if is_isolated:
                tau_c *= (1.0 - self.config.suppression_alpha)
                suppression_applied = True
            
            # 综合分数
            score = r_c * tau_c * conf_c + self.config.score_eps
            
            # 4. 阈值判定
            thresh = self.config.category_thresholds.get(
                cls_id, CategoryThreshold()
            )
            
            n_det_frames = len(set(det_indices))
            passed = (
                score >= thresh.min_score and
                r_c >= thresh.min_frame_ratio and
                tau_c >= thresh.min_temporal_consistency and
                conf_c >= thresh.min_avg_confidence and
                n_det_frames >= thresh.min_det_frames
            )
            
            event = VideoEventResult(
                cls_id=cls_id,
                name_cn=CLASS_NAMES.get(cls_id, f'class_{cls_id}'),
                name_en=CLASS_NAMES_EN.get(cls_id, f'class_{cls_id}'),
                score=round(score, 6),
                frame_ratio=round(r_c, 4),
                temporal_consistency=round(tau_c, 4),
                avg_confidence=round(conf_c, 4),
                det_frames=n_det_frames,
                total_frames=total_frames,
                passed=passed,
                suppression_applied=suppression_applied,
            )
            all_events.append(event)
        
        # 5. 按分数排序, 选出主事件
        all_events.sort(key=lambda e: e.score, reverse=True)
        primary_event = None
        for event in all_events:
            if event.passed:
                primary_event = event
                break
        
        # 转换 timeline 为普通 dict
        timeline_dict = {
            sec: dict(cls_counts) for sec, cls_counts in sorted(timeline.items())
        }
        
        return VideoDecisionResult(
            primary_event=primary_event,
            all_events=all_events,
            total_frames=total_frames,
            duration_sec=round(duration_sec, 1),
            timeline=timeline_dict,
        )
    
    def decide_batch(
        self,
        batch_detections: Dict[str, List[FrameDetection]],
        batch_total_frames: Dict[str, int],
        batch_fps: Optional[Dict[str, float]] = None,
    ) -> Dict[str, VideoDecisionResult]:
        """
        批量视频级决策
        
        Args:
            batch_detections: {video_name: [FrameDetection, ...]}
            batch_total_frames: {video_name: total_frames}
            batch_fps: {video_name: fps} (可选)
        
        Returns:
            {video_name: VideoDecisionResult}
        """
        results = {}
        for vname in batch_detections:
            fps = (batch_fps or {}).get(vname, self.config.fps)
            results[vname] = self.decide(
                batch_detections[vname],
                batch_total_frames[vname],
                fps=fps,
            )
        return results
    
    def format_report(self, result: VideoDecisionResult, video_name: str = "") -> str:
        """
        格式化输出视频级判定报告
        
        Args:
            result: VideoDecisionResult
            video_name: 视频文件名 (可选)
        
        Returns:
            格式化的报告字符串
        """
        lines = []
        lines.append("=" * 65)
        lines.append(f"  TVAD 视频级事件判定报告")
        if video_name:
            lines.append(f"  视频: {video_name}")
        lines.append(f"  总帧数: {result.total_frames}  |  时长: {result.duration_sec}s")
        lines.append("=" * 65)
        
        if result.primary_event:
            pe = result.primary_event
            lines.append(f"  ★ 主事件: {pe.name_cn} ({pe.name_en})")
            lines.append(f"    综合分数 S = {pe.score:.6f}")
            lines.append(f"    帧占比  r = {pe.frame_ratio:.4f} "
                        f"({pe.det_frames}/{pe.total_frames} 帧)")
            lines.append(f"    时序一致 τ = {pe.temporal_consistency:.4f}")
            lines.append(f"    平均置信 c = {pe.avg_confidence:.4f}")
            if pe.suppression_applied:
                lines.append(f"    ⚠ 已触发孤立帧抑制")
        else:
            lines.append("  ★ 主事件: 无 (所有类别均未通过阈值)")
        
        lines.append("-" * 65)
        lines.append("  各类别详细评估:")
        lines.append(f"  {'类别':<12} {'分数':>10} {'帧占比':>8} "
                    f"{'时序τ':>8} {'置信度':>8} {'通过':>6}")
        lines.append("-" * 65)
        
        for ev in result.all_events:
            passed_str = "✓" if ev.passed else "✗"
            supp_str = " [抑制]" if ev.suppression_applied else ""
            lines.append(
                f"  {ev.name_cn:<10} {ev.score:>10.6f} {ev.frame_ratio:>8.4f} "
                f"{ev.temporal_consistency:>8.4f} {ev.avg_confidence:>8.4f} "
                f"{passed_str:>6}{supp_str}"
            )
        
        lines.append("=" * 65)
        return "\n".join(lines)
    
    def to_dict(self, result: VideoDecisionResult) -> dict:
        """
        将结果转换为可 JSON 序列化的字典
        
        Args:
            result: VideoDecisionResult
        
        Returns:
            dict
        """
        def event_to_dict(ev: VideoEventResult) -> dict:
            return {
                'cls_id': ev.cls_id,
                'name_cn': ev.name_cn,
                'name_en': ev.name_en,
                'score': ev.score,
                'frame_ratio': ev.frame_ratio,
                'temporal_consistency': ev.temporal_consistency,
                'avg_confidence': ev.avg_confidence,
                'det_frames': ev.det_frames,
                'total_frames': ev.total_frames,
                'passed': ev.passed,
                'suppression_applied': ev.suppression_applied,
            }
        
        return {
            'primary_event': (
                event_to_dict(result.primary_event) 
                if result.primary_event else None
            ),
            'all_events': [event_to_dict(ev) for ev in result.all_events],
            'total_frames': result.total_frames,
            'duration_sec': result.duration_sec,
            'timeline': {
                str(k): {str(ck): cv for ck, cv in v.items()} 
                for k, v in result.timeline.items()
            },
        }


# ============ 便捷工厂函数 ============

def create_tvad(
    temporal_window: float = 2.0,
    suppression_alpha: float = 0.3,
    **kwargs
) -> TVAD:
    """
    创建 TVAD 实例的便捷函数
    
    Args:
        temporal_window: 滑窗大小 (秒)
        suppression_alpha: 孤立帧抑制因子
        **kwargs: 其他 VideoDecisionConfig 参数
    
    Returns:
        TVAD 实例
    """
    config = VideoDecisionConfig(
        temporal_window=temporal_window,
        suppression_alpha=suppression_alpha,
        **kwargs,
    )
    return TVAD(config)


# ============ 测试 ============

if __name__ == "__main__":
    print("=" * 60)
    print("  TVAD 模块功能测试")
    print("=" * 60)
    
    tvad = TVAD()
    
    # 模拟场景: 100帧视频, 25fps, 共4秒
    total_frames = 100
    fps = 25.0
    
    # 模拟检测结果: 抛洒物在 frame 10-80 持续检出, 逆行仅在 frame 50-53 出现
    detections = []
    
    # 抛洒物: 持续性检出 (应该通过)
    for fi in range(10, 80, 2):
        detections.append(FrameDetection(
            frame_idx=fi, cls_id=0, confidence=0.75 + 0.1 * (fi % 3)
        ))
    
    # 逆行: 孤立检出 (应该被抑制)
    for fi in [50, 51, 52]:
        detections.append(FrameDetection(
            frame_idx=fi, cls_id=2, confidence=0.60
        ))
    
    result = tvad.decide(detections, total_frames, fps=fps)
    
    # 打印报告
    report = tvad.format_report(result, "test_video.mp4")
    print(report)
    
    # 验证
    assert result.primary_event is not None, "应检出主事件"
    assert result.primary_event.cls_id == 0, "主事件应为抛洒物"
    assert result.primary_event.passed, "抛洒物应通过阈值"
    
    # 测试空视频
    empty_result = tvad.decide([], 50, fps=25.0)
    assert empty_result.primary_event is None, "空视频应无主事件"
    
    # 测试批量
    batch_dets = {"v1.mp4": detections, "v2.mp4": []}
    batch_frames = {"v1.mp4": 100, "v2.mp4": 50}
    batch_results = tvad.decide_batch(batch_dets, batch_frames)
    assert len(batch_results) == 2
    
    # 测试序列化
    result_dict = tvad.to_dict(result)
    import json
    json_str = json.dumps(result_dict, ensure_ascii=False, indent=2)
    assert '"score"' in json_str
    
    print("\n✅ TVAD module all tests passed!")
