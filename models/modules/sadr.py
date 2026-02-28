"""
SADR: Scale-Adaptive Dynamic Routing Module (创新点1)
尺度自适应动态路由模块

核心创新:
1. 逐像素尺度路由 —— 每个空间位置独立选择最优感受野组合
2. 条带池化驱动预测 —— 利用道路场景方向性先验
3. 统一膨胀卷积 —— 等参数量覆盖更大感受野范围

改进点 (相对于方案文档):
- 增加了通道注意力分支对尺度预测的辅助 (Channel-Spatial Joint Routing)
- DilatedBranch 增加了 SE 通道校准, 提升分支内特征质量
- 支持可配置的膨胀率列表, 更灵活
- forward 中使用 torch.stack + einsum 替代循环求和, 提升效率
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class StripPooling(nn.Module):
    """
    条带池化模块
    分别沿水平和垂直方向进行全局平均池化，
    保留方向性空间结构信息（对道路场景至关重要）
    
    改进: 增加了可选的通道压缩比参数 reduction
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 32)
        
        # 水平条带: (B,C,H,W) → (B,C,H,1) → 1×1Conv
        self.h_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.h_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        # 垂直条带: (B,C,H,W) → (B,C,1,W) → 1×1Conv
        self.v_pool = nn.AdaptiveAvgPool2d((1, None))
        self.v_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        # 融合层
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        self.mid_channels = mid_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 水平条带: (B, mid, H, 1) → 自动广播
        h_feat = self.h_conv(self.h_pool(x))
        # 垂直条带: (B, mid, 1, W) → 自动广播
        v_feat = self.v_conv(self.v_pool(x))
        # 广播相加 → (B, mid, H, W)
        combined = h_feat + v_feat
        return self.fuse(combined)


class ChannelGate(nn.Module):
    """
    轻量级通道注意力门控 (改进补充)
    用于辅助尺度预测, 提供全局通道统计信息
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.mlp(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ScalePredictor(nn.Module):
    """
    尺度预测子网络
    为每个空间位置预测各尺度分支的亲和力权重
    
    改进: 融合条带池化(空间)和通道门控(通道)双路径信息
    """
    def __init__(self, in_channels: int, num_branches: int = 3):
        super().__init__()
        self.strip_pool = StripPooling(in_channels)
        mid = self.strip_pool.mid_channels
        
        # 通道统计辅助: 提供全局上下文
        self.channel_gate = ChannelGate(in_channels, reduction=16)
        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True)
        )
        
        # 融合后预测尺度权重
        self.predictor = nn.Sequential(
            nn.Conv2d(mid * 2, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, num_branches, 1, bias=True)
        )
        self.num_branches = num_branches
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 条带池化空间特征
        strip_feat = self.strip_pool(x)             # (B, mid, H, W)
        # 通道门控特征
        ch_feat = self.channel_proj(self.channel_gate(x))  # (B, mid, H, W)
        # 拼接
        combined = torch.cat([strip_feat, ch_feat], dim=1)  # (B, 2*mid, H, W)
        # 预测
        logits = self.predictor(combined)            # (B, num_branches, H, W)
        weights = F.softmax(logits, dim=1)
        return weights


class DilatedBranch(nn.Module):
    """
    膨胀卷积分支: 深度可分离膨胀卷积 + 通道校准
    
    改进: 增加了轻量SE通道校准, 让每个分支自适应调整通道响应
    """
    def __init__(self, channels: int, dilation: int, use_se: bool = True):
        super().__init__()
        # 深度膨胀卷积
        self.dw_conv = nn.Conv2d(
            channels, channels, 3,
            padding=dilation, dilation=dilation,
            groups=channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(channels)
        
        # 逐点卷积
        self.pw_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(channels)
        
        self.act = nn.SiLU(inplace=True)
        
        # 可选 SE 通道校准
        self.use_se = use_se
        if use_se:
            se_mid = max(channels // 16, 8)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, se_mid, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_mid, channels, 1, bias=False),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.dw_bn(self.dw_conv(x)))
        out = self.act(self.pw_bn(self.pw_conv(out)))
        if self.use_se:
            out = out * self.se(out)
        return out


class SADR(nn.Module):
    """
    Scale-Adaptive Dynamic Routing Module
    尺度自适应动态路由模块
    
    输入: (B, C, H, W) 特征图
    输出: (B, C, H, W) 特征图 (尺度自适应增强后)
    
    Args:
        channels (int): 输入/输出通道数
        dilations (list): 各分支膨胀率列表, 默认 [1, 3, 5]
            - d=1: 感受野 3×3  (小目标 —— 抛洒物)
            - d=3: 感受野 7×7  (中目标)
            - d=5: 感受野 11×11 (大目标 —— 违停车辆)
        use_se (bool): 是否在分支内使用SE通道校准
    
    工作原理:
    1. 多个膨胀卷积分支提取不同感受野特征
    2. 尺度预测网络为每个像素位置生成路由权重
    3. 逐像素加权融合 + 残差连接
    """
    def __init__(self, channels: int = 0, dilations: Optional[List[int]] = None, 
                 use_se: bool = True):
        super().__init__()
        if dilations is None:
            dilations = [1, 3, 5]
        
        # 保存配置参数 (用于延迟构建)
        self._dilations = dilations
        self._use_se = use_se
        self.num_branches = len(dilations)
        
        # 可学习残差缩放因子 (不依赖通道数, 提前创建)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 始终延迟构建: Ultralytics 的 width_multiple 会缩放实际通道数,
        # 但 YAML 中的参数值不会随之变化, 因此不能用 YAML 传入的 channels
        # 构建子模块, 必须在首次 forward 时从实际输入 tensor 推断通道数.
        self._built = False
    
    def _build(self, channels: int):
        """根据实际通道数构建所有子模块 (支持延迟构建)"""
        # 多尺度膨胀卷积分支
        self.branches = nn.ModuleList([
            DilatedBranch(channels, d, use_se=self._use_se)
            for d in self._dilations
        ])
        
        # 尺度预测子网络
        self.scale_predictor = ScalePredictor(channels, self.num_branches)
        
        # 输出投影 + 残差
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self._built = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 延迟构建: 首次 forward 时根据实际输入通道数构建子模块
        if not self._built:
            self._build(x.shape[1])
            self.to(x.device)
        
        # 1. 计算各分支特征 → stack 成 (B, num_branches, C, H, W)
        branch_feats = torch.stack(
            [branch(x) for branch in self.branches], dim=1
        )
        
        # 2. 预测逐像素尺度路由权重: (B, num_branches, H, W)
        weights = self.scale_predictor(x)
        
        # 3. 高效加权融合: einsum 替代循环
        # weights: (B, K, H, W) → (B, K, 1, H, W) 广播
        # branch_feats: (B, K, C, H, W)
        fused = torch.einsum('bkhw,bkchw->bchw', weights, branch_feats)
        
        # 4. 投影 + 带缩放因子的残差连接
        return x + self.gamma * self.proj(fused)


# ============ 消融实验用的变体 ============

class SADR_NoStripPool(SADR):
    """
    消融变体: 将条带池化替换为普通全局池化
    用于验证条带池化的有效性
    """
    def _build(self, channels):
        """构建后替换条带池化为全局池化"""
        super()._build(channels)
        mid = self.scale_predictor.strip_pool.mid_channels
        self.scale_predictor.strip_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=1)  # placeholder, forward 时会被替换
        )


if __name__ == "__main__":
    # 快速测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试不同通道数
    for c in [256, 512, 1024]:
        x = torch.randn(2, c, 20, 20).to(device)
        module = SADR(c).to(device)
        out = module(x)
        params = sum(p.numel() for p in module.parameters())
        print(f"SADR(channels={c}): input={x.shape} → output={out.shape}, "
              f"params={params/1e3:.1f}K")
    
    print("\n✅ SADR module test passed!")
