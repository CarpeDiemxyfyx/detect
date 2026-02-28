"""
自定义模块包
包含三个创新模块:
1. SADR - 尺度自适应动态路由模块 (Scale-Adaptive Dynamic Routing)
2. BDFR - 背景解耦特征精炼模块 (Background-Decoupled Feature Refinement)
3. TVAD - 时序感知视频聚合决策模块 (Temporal-aware Video Aggregation Decision)
"""

from .sadr import SADR, StripPooling, ScalePredictor, DilatedBranch
from .bdfr import BDFR
from .tvad import TVAD, VideoDecisionConfig, FrameDetection, VideoDecisionResult, create_tvad

__all__ = [
    'SADR', 'BDFR', 'TVAD',
    'StripPooling', 'ScalePredictor', 'DilatedBranch',
    'VideoDecisionConfig', 'FrameDetection', 'VideoDecisionResult', 'create_tvad',
]
