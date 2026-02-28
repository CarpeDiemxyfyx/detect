"""
BDFR: Background-Decoupled Feature Refinement Module (创新点2)
背景解耦特征精炼模块

核心创新:
1. 可学习背景原型向量 —— 端到端学习道路背景模式
2. 偏离度驱动注意力 —— 基于"与背景的距离"而非统计量
3. EMA原型稳定更新 —— 防止训练震荡

改进点 (相对于方案文档):
- 增加了温度参数 tau 控制偏离度映射的锐利度
- 偏离度映射网络使用深度可分离卷积替代标准卷积, 减少参数
- 增加了前景原型 (可选), 实现前景-背景双向解耦
- 原型初始化使用 Xavier 均匀分布, 更利于收敛
- 增加了偏离度图可视化接口 get_deviation_map(), 方便论文分析
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 (轻量化)"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                 padding: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, 
                           groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


class DeviationAttentionHead(nn.Module):
    """
    偏离度→注意力映射头
    使用深度可分离卷积, 比原方案中的标准卷积更轻量
    增加多尺度感知: 3×3 + 5×5 双路径
    """
    def __init__(self, in_channels: int = 1, mid_channels: int = 16):
        super().__init__()
        # 3×3 路径
        self.branch_small = DepthwiseSeparableConv(in_channels, mid_channels // 2, 3, 1)
        # 5×5 路径 (用于捕获更大区域的偏离模式)
        self.branch_large = DepthwiseSeparableConv(in_channels, mid_channels // 2, 5, 2)
        # 融合输出
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels, 1, 1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.branch_small(x)
        f2 = self.branch_large(x)
        return self.fuse(torch.cat([f1, f2], dim=1))


class BDFR(nn.Module):
    """
    Background-Decoupled Feature Refinement Module
    背景解耦特征精炼模块
    
    Args:
        channels (int): 输入特征通道数
        num_prototypes (int): 背景原型数量 K (默认8, 表示8种背景模式)
        proj_dim (int|None): 投影维度 (默认 channels//4, 降维减少距离计算开销)
        ema_momentum (float): EMA动量系数 (默认0.996)
        tau (float): 温度参数, 控制偏离度映射的对比度 (默认1.0)
        use_fg_proto (bool): 是否使用前景原型进行双向解耦 (默认False)
    
    输入: (B, C, H, W)
    输出: (B, C, H, W)
    """
    def __init__(self, channels: int = 0, num_prototypes: int = 8, 
                 proj_dim: Optional[int] = None, ema_momentum: float = 0.996,
                 tau: float = 1.0, use_fg_proto: bool = False):
        super().__init__()
        # 保存配置参数 (用于延迟构建)
        self._num_prototypes = num_prototypes
        self._proj_dim_cfg = proj_dim     # 用户指定值, 可能为 None
        self._ema_momentum = ema_momentum
        self._tau = tau
        self._use_fg_proto = use_fg_proto
        
        # 可学习残差缩放 (不依赖通道数, 提前创建)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 用于可视化的缓存
        self._last_deviation_map = None
        
        # 始终延迟构建: Ultralytics 的 width_multiple 会缩放实际通道数,
        # 但 YAML 中的参数值不会随之变化, 因此不能用 YAML 传入的 channels
        # 构建子模块, 必须在首次 forward 时从实际输入 tensor 推断通道数.
        self._built = False
    
    def _build(self, channels: int):
        """根据实际通道数构建所有子模块 (支持延迟构建)"""
        self.channels = channels
        self.num_prototypes = self._num_prototypes
        self.ema_momentum = self._ema_momentum
        self.tau = self._tau
        self.use_fg_proto = self._use_fg_proto
        self.proj_dim = self._proj_dim_cfg or max(channels // 4, 32)
        
        # ===== 特征投影层 =====
        self.feature_proj = nn.Sequential(
            nn.Conv2d(channels, self.proj_dim, 1, bias=False),
            nn.BatchNorm2d(self.proj_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.proj_dim, self.proj_dim, 3, padding=1, 
                     groups=self.proj_dim, bias=False),
            nn.BatchNorm2d(self.proj_dim),
            nn.SiLU(inplace=True),
        )
        
        # ===== 可学习背景原型 =====
        self.bg_prototypes = nn.Parameter(
            torch.empty(self.num_prototypes, self.proj_dim)
        )
        nn.init.xavier_uniform_(self.bg_prototypes)
        
        # EMA 缓冲
        self.register_buffer(
            'bg_proto_ema',
            torch.zeros(self.num_prototypes, self.proj_dim)
        )
        self.register_buffer('ema_initialized', torch.tensor(False))
        
        # ===== 可选前景原型 =====
        if self.use_fg_proto:
            self.fg_prototypes = nn.Parameter(
                torch.empty(self.num_prototypes // 2, self.proj_dim)
            )
            nn.init.xavier_uniform_(self.fg_prototypes)
        
        # ===== 偏离度 → 注意力映射 =====
        dev_in_ch = 2 if self.use_fg_proto else 1
        self.deviation_to_attn = DeviationAttentionHead(dev_in_ch, 16)
        
        self._built = True
    
    @torch.no_grad()
    def _init_ema(self, features: torch.Tensor):
        """首次调用时用当前特征初始化EMA原型"""
        if not self.ema_initialized:
            self.bg_proto_ema.copy_(self.bg_prototypes.data)
            self.ema_initialized.fill_(True)
    
    @torch.no_grad()
    def _update_prototypes_ema(self, features: torch.Tensor):
        """
        EMA更新背景原型（仅在训练时调用）
        
        改进: 
        - 增加了首次初始化逻辑
        - 使用 cosine similarity 替代 L2 距离进行分配, 更稳定
        - 增加了空聚类保护
        """
        if not self.training:
            return
        
        self._init_ema(features)
        
        B, C, H, W = features.shape
        feat_flat = features.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
        
        # L2 归一化后用余弦相似度分配
        feat_norm = F.normalize(feat_flat, dim=1)
        proto_norm = F.normalize(self.bg_prototypes.data, dim=1)
        sim = torch.mm(feat_norm, proto_norm.t())  # (N, K)
        assignments = sim.argmax(dim=1)
        
        for k in range(self.num_prototypes):
            mask = (assignments == k)
            if mask.sum() > 10:  # 至少10个样本才更新, 避免噪声
                cluster_mean = feat_flat[mask].mean(dim=0)
                self.bg_proto_ema[k] = (
                    self.ema_momentum * self.bg_proto_ema[k] +
                    (1 - self.ema_momentum) * cluster_mean
                )
        
        # 软同步回可学习参数
        self.bg_prototypes.data = (
            0.95 * self.bg_prototypes.data + 
            0.05 * self.bg_proto_ema
        )
    
    def _compute_deviation(self, proj_features: torch.Tensor) -> torch.Tensor:
        """
        计算每个空间位置到最近背景原型的偏离度
        
        改进: 增加温度参数 tau 控制偏离度的对比度
        
        Returns:
            deviation_map: (B, 1, H, W) 归一化偏离度图, 值域 [0, 1]
        """
        B, C, H, W = proj_features.shape
        feat = proj_features.flatten(2).permute(0, 2, 1)  # (B, N, C)
        protos = self.bg_prototypes.unsqueeze(0)            # (1, K, C)
        
        # L2 距离: (B, N, K)
        dist = torch.cdist(feat, protos)
        
        # 最小距离 (到最近原型): (B, N)
        min_dist, _ = dist.min(dim=2)
        
        # 温度缩放 + 归一化
        min_dist = min_dist / (self.tau + 1e-6)
        d_min = min_dist.amin(dim=1, keepdim=True)
        d_max = min_dist.amax(dim=1, keepdim=True)
        deviation = (min_dist - d_min) / (d_max - d_min + 1e-6)
        
        return deviation.view(B, 1, H, W)
    
    def _compute_fg_affinity(self, proj_features: torch.Tensor) -> torch.Tensor:
        """
        (可选) 计算前景亲和度
        偏离度高 + 前景亲和高 → 更强的异常响应
        """
        if not self.use_fg_proto:
            return None
        
        B, C, H, W = proj_features.shape
        feat = proj_features.flatten(2).permute(0, 2, 1)
        fg_protos = self.fg_prototypes.unsqueeze(0)
        
        # 余弦相似度
        feat_norm = F.normalize(feat, dim=2)
        fg_norm = F.normalize(fg_protos, dim=2)
        sim = torch.bmm(feat_norm, fg_norm.expand(B, -1, -1).permute(0, 2, 1))
        max_sim, _ = sim.max(dim=2)  # (B, N)
        
        # 归一化到 [0, 1]
        s_min = max_sim.amin(dim=1, keepdim=True)
        s_max = max_sim.amax(dim=1, keepdim=True)
        affinity = (max_sim - s_min) / (s_max - s_min + 1e-6)
        
        return affinity.view(B, 1, H, W)
    
    def get_deviation_map(self) -> Optional[torch.Tensor]:
        """获取最近一次前向传播的偏离度图 (用于可视化)"""
        return self._last_deviation_map
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 输入特征
        Returns:
            (B, C, H, W) 精炼后的特征
        """
        # 延迟构建: 首次 forward 时根据实际输入通道数构建子模块
        if not self._built:
            self._build(x.shape[1])
            self.to(x.device)
        
        # 1. 特征投影
        proj = self.feature_proj(x)  # (B, proj_dim, H, W)
        
        # 2. EMA更新原型 (仅训练时)
        if self.training:
            self._update_prototypes_ema(proj.detach())
        
        # 3. 计算偏离度图
        deviation_map = self._compute_deviation(proj)
        self._last_deviation_map = deviation_map.detach()
        
        # 4. 构建注意力输入
        if self.use_fg_proto:
            fg_affinity = self._compute_fg_affinity(proj)
            attn_input = torch.cat([deviation_map, fg_affinity], dim=1)
        else:
            attn_input = deviation_map
        
        # 5. 偏离度 → 注意力权重
        attn = self.deviation_to_attn(attn_input)  # (B, 1, H, W)
        
        # 6. 注意力加权 + 带缩放因子的残差
        # Y = X + γ(X⊙A): 初始 γ=0 时为恒等映射, 随训练逐步引入注意力调制
        return x + self.gamma * (x * attn)


# ============ 消融实验用的变体 ============

class BDFR_NoEMA(BDFR):
    """消融变体: 去掉EMA更新, 仅依赖梯度更新原型"""
    @torch.no_grad()
    def _update_prototypes_ema(self, features):
        pass  # 不执行EMA更新


class BDFR_FixedProto(BDFR):
    """消融变体: 固定原型 (不学习), 使用随机初始化"""
    def _build(self, channels):
        super()._build(channels)
        self.bg_prototypes.requires_grad = False


if __name__ == "__main__":
    # 快速测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for c in [256, 512, 1024]:
        x = torch.randn(2, c, 20, 20).to(device)
        
        # 标准版
        module = BDFR(c).to(device)
        out = module(x)
        params = sum(p.numel() for p in module.parameters())
        print(f"BDFR(channels={c}): input={x.shape} → output={out.shape}, "
              f"params={params/1e3:.1f}K")
        
        # 验证偏离度图
        dev_map = module.get_deviation_map()
        if dev_map is not None:
            print(f"  deviation_map: {dev_map.shape}, "
                  f"range=[{dev_map.min():.3f}, {dev_map.max():.3f}]")
    
    # 测试前景原型变体
    x = torch.randn(2, 512, 20, 20).to(device)
    module_fg = BDFR(512, use_fg_proto=True).to(device)
    out_fg = module_fg(x)
    print(f"\nBDFR with fg_proto: {x.shape} → {out_fg.shape}")
    
    print("\n✅ BDFR module test passed!")
