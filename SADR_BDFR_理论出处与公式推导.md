# SADR 与 BDFR 创新模块 —— 理论出处、公式推导与参考文献

> 本文档详细梳理本项目两个核心创新模块的**理论基础、数学公式推导、各子组件的学术出处**，供论文撰写参考。

---

## 一、SADR：尺度自适应动态路由模块 (Scale-Adaptive Dynamic Routing)

### 1.1 模块总览

SADR 的核心思想是：**为特征图的每个空间位置 $(h, w)$ 独立预测尺度路由权重，在多个不同感受野分支之间进行逐像素加权融合**。

该模块并非从单一论文直接迁移而来，而是融合了以下多个经典方法的核心思想，并在此基础上进行了**面向道路场景的创新组合与改进**：

---

### 1.2 组件出处与公式

#### 1.2.1 条带池化 (Strip Pooling) —— 出自 SPNet

**出处**：

> **Hou Q, Zhang L, Cheng M M, et al.** Strip Pooling: Rethinking Spatial Pooling for Scene Parsing[C]. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020: 4003-4012.

**核心思想**：传统全局平均池化 (GAP) 将整个特征图压缩为一个向量，完全丢失空间结构信息。Strip Pooling 分别沿水平和垂直方向进行条带状池化，保留方向性空间分布信息。

**公式推导**：

设输入特征图为 $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$，则：

**水平条带池化**（沿宽度方向聚合，保留高度维度信息）：

$$
\mathbf{F}_h(c, i, 1) = \frac{1}{W} \sum_{j=1}^{W} \mathbf{F}(c, i, j), \quad \mathbf{F}_h \in \mathbb{R}^{C \times H \times 1}
$$

**垂直条带池化**（沿高度方向聚合，保留宽度维度信息）：

$$
\mathbf{F}_v(c, 1, j) = \frac{1}{H} \sum_{i=1}^{H} \mathbf{F}(c, i, j), \quad \mathbf{F}_v \in \mathbb{R}^{C \times 1 \times W}
$$

经过 $1 \times 1$ 卷积降维至 $d$ 维后，通过广播相加融合：

$$
\mathbf{F}_{\text{strip}}(i, j) = \phi\left( \text{Conv}_{1 \times 1}(\mathbf{F}_h(i)) + \text{Conv}_{1 \times 1}(\mathbf{F}_v(j)) \right) \in \mathbb{R}^{d \times H \times W}
$$

其中 $\phi$ 为 SiLU 激活函数。

**本文改进**：原始 SPNet 将条带池化用于场景分割的上下文建模，本文将其用于**尺度预测网络**中，利用道路场景的方向性先验（道路纵向延伸，目标尺度沿纵向近大远小）来预测空间自适应的尺度路由权重。复杂度仅为 $O(C \cdot H + C \cdot W)$，远低于自注意力机制的 $O(H^2 W^2)$。

---

#### 1.2.2 膨胀卷积多分支 (Dilated Convolution Branches) —— 出自 DeepLab / ASPP

**出处**：

> **Chen L C, Papandreou G, Kokkinos I, et al.** DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs[J]. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 2018, 40(4): 834-848.

> **Chen L C, Zhu Y, Papandreou G, et al.** Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation[C]. *European Conference on Computer Vision (ECCV)*, 2018: 801-818.

**核心思想**：使用不同膨胀率 (dilation rate) 的卷积核，在不增加参数量的前提下扩大感受野。

**公式推导**：

标准 $3 \times 3$ 卷积的感受野为 $3 \times 3$。膨胀率为 $r$ 的膨胀卷积，其有效感受野为：

$$
\text{RF}(r) = (k - 1) \times r + 1
$$

其中 $k=3$ 为卷积核大小。对于本文采用的三个分支：

| 分支 | 膨胀率 $r$ | 有效感受野 | 适用目标尺度 |
|------|-----------|-----------|------------|
| Branch-S | $r=1$ | $3 \times 3$ | 小目标（抛洒物，8~32px） |
| Branch-M | $r=3$ | $7 \times 7$ | 中目标（违停车辆，64~256px） |
| Branch-L | $r=5$ | $11 \times 11$ | 大目标（逆行车辆，>256px） |

每个分支采用**深度可分离膨胀卷积**（出自 MobileNet [Howard et al., 2017]），设输入和输出通道数均为 $C$（即 $C_{\text{in}} = C_{\text{out}} = C$），其参数量为：

$$
\text{Params}_{\text{DW-Dilated}} = \underbrace{C_{\text{in}} \times k^2}_{\text{深度卷积}} + \underbrace{C_{\text{in}} \times C_{\text{out}}}_{\text{逐点卷积}} = C \times k^2 + C^2
$$

当 $k=3$ 时，$\text{Params} = 9C + C^2$。相比标准卷积的 $C_{\text{in}} \times C_{\text{out}} \times k^2 = 9C^2$，参数量降低为原来的 $\frac{9C + C^2}{9C^2} = \frac{1}{C} + \frac{1}{9}$，当 $C$ 较大时近似降低至 $\frac{1}{9}$。

**本文改进**：与 ASPP 直接将各分支输出拼接或求和不同，本文通过**尺度预测网络**生成逐像素的软路由权重，实现空间自适应融合（详见 1.2.4）。

---

#### 1.2.3 通道注意力门控 (Channel Gate / SE Attention) —— 出自 SENet

**出处**：

> **Hu J, Shen L, Sun G.** Squeeze-and-Excitation Networks[C]. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018: 7132-7141.

**核心思想**：通过全局平均池化→全连接瓶颈→Sigmoid 的路径，为每个通道生成自适应缩放权重。

**公式推导**：

$$
\mathbf{z} = \text{GAP}(\mathbf{F}) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{F}(:, i, j) \in \mathbb{R}^{C}
$$

$$
\mathbf{s} = \sigma\left( \mathbf{W}_2 \cdot \phi\left( \mathbf{W}_1 \cdot \mathbf{z} \right) \right) \in \mathbb{R}^{C}
$$

$$
\tilde{\mathbf{F}}(c, :, :) = \mathbf{s}(c) \cdot \mathbf{F}(c, :, :)
$$

其中 $\mathbf{W}_1 \in \mathbb{R}^{(C/r) \times C}$，$\mathbf{W}_2 \in \mathbb{R}^{C \times (C/r)}$，$r$ 为压缩比（本文取 $r=16$），$\phi$ 为 SiLU 激活，$\sigma$ 为 Sigmoid 函数。

**本文改进**：SE 注意力在 SADR 中有两处应用：
1. **ScalePredictor 中的 ChannelGate**：与条带池化并行，提供全局通道统计信息，辅助尺度预测（"Channel-Spatial Joint Routing"）；
2. **DilatedBranch 中的 SE 校准**：在每个膨胀卷积分支内部使用 SE 模块校准通道响应，提升单分支特征质量。

---

#### 1.2.4 动态路由 / 条件计算 (Dynamic Routing) —— 出自 CondConv / SKNet

**出处**：

> **Li X, Wang W, Hu X, et al.** Selective Kernel Networks[C]. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019: 510-519.

> **Yang B, Bender G, Le Q V, et al.** CondConv: Conditionally Parameterized Convolutions for Efficient Inference[C]. *Advances in Neural Information Processing Systems (NeurIPS)*, 2019: 1307-1318.

> **Chen Y, Dai X, Liu M, et al.** Dynamic Convolution: Attention over Convolution Kernels[C]. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020: 11030-11039.

**SKNet 核心思想**：通过 Split-Fuse-Select 机制，在不同核大小的分支之间进行**通道级别**的动态选择。

**SKNet 公式**：

$$
\mathbf{a}_c = \frac{\exp(\mathbf{W}^A_c \cdot \mathbf{z})}{\exp(\mathbf{W}^A_c \cdot \mathbf{z}) + \exp(\mathbf{W}^B_c \cdot \mathbf{z})}, \quad \mathbf{z} = \text{GAP}(\mathbf{U}^A + \mathbf{U}^B)
$$

其中 $\mathbf{a}_c$ 是第 $c$ 个通道在分支 A 上的选择权重，**所有空间位置共享同一权重**（全图仅一个向量）。

**本文 SADR 的关键改进 —— 逐像素尺度路由**：

SADR 将路由粒度从 SKNet 的**通道级（全图共享）** 提升到**像素级（逐位置独立）**：

$$
\mathbf{W}(b, :, h, w) = \text{Softmax}\left( g_{\theta}\left( \mathbf{F}_{\text{in}} \right) \right)(b, :, h, w) \in \mathbb{R}^{K}
$$

其中 $g_{\theta}$ 为尺度预测子网络（融合条带池化空间特征与通道门控特征），$K=3$ 为分支数量。

**逐像素加权融合公式**：

$$
\mathbf{F}_{\text{fused}}(b, c, h, w) = \sum_{k=1}^{K} \mathbf{W}(b, k, h, w) \cdot \mathbf{F}_k(b, c, h, w)
$$

其中 $\mathbf{F}_k$ 为第 $k$ 个膨胀卷积分支的输出，$\mathbf{W}(b, k, h, w)$ 为空间位置 $(h, w)$ 在第 $k$ 个分支上的软路由权重。

使用 Einstein Summation 高效实现：

$$
\mathbf{F}_{\text{fused}} = \texttt{einsum}(\texttt{`bkhw, bkchw -> bchw`},\; \mathbf{W},\; \mathbf{F}_{\text{stack}})
$$

**最终输出**（带可学习残差缩放）：

$$
\mathbf{F}_{\text{out}} = \mathbf{F}_{\text{in}} + \gamma \cdot \text{Conv}_{1 \times 1}\left( \mathbf{F}_{\text{fused}} \right)
$$

其中 $\gamma$ 初始化为 $0$，在训练初期使模块近似恒等映射，稳定训练过程（该技巧出自 LayerScale [Touvron et al., ICCV 2021]）。

---

#### 1.2.5 可学习残差缩放 (Learnable Residual Scaling / LayerScale)

**出处**：

> **Touvron H, Cord M, Sablayrolles A, et al.** Going Deeper with Image Transformers[C]. *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021: 32-42.

**公式**：

$$
\mathbf{y} = \mathbf{x} + \gamma \cdot f(\mathbf{x}), \quad \gamma \in \mathbb{R}, \quad \gamma_0 = 0
$$

$\gamma$ 从零开始学习，确保训练初期新增模块不会破坏预训练特征流。

---

### 1.3 SADR 完整前向传播公式汇总

设输入为 $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$：

**Step 1：多尺度特征提取**

$$
\mathbf{F}_k = \text{DilatedBranch}_k(\mathbf{X}), \quad k \in \{1, 2, 3\}, \quad r_k \in \{1, 3, 5\}
$$

$$
\text{DilatedBranch}_k(\mathbf{X}) = \text{SE}\left(\text{PW-Conv}\left(\text{DW-DilConv}_{r_k}(\mathbf{X})\right)\right)
$$

**Step 2：尺度路由权重预测（通道-空间联合路由）**

尺度预测子网络包含两条并行路径，分别捕获**空间方向性**和**通道全局统计**信息：

**路径 A — 条带池化空间路径**：利用水平/垂直条带池化获取道路方向性先验，输出保留完整空间分辨率的中间特征：

$$
\mathbf{F}_{\text{strip}} = \text{StripPooling}(\mathbf{X}) \in \mathbb{R}^{B \times d \times H \times W}
$$

**路径 B — 通道门控路径**：通过 SE 风格的通道注意力（ChannelGate）对输入进行全局通道重标定，再用 $1\times1$ 卷积投影到与条带池化相同的维度 $d$。该路径提供「哪些通道在当前输入中更活跃」的全局统计信息，弥补条带池化仅关注空间分布的不足：

$$
\mathbf{F}_{\text{ch}} = \text{Proj}_{1\times1}\!\left(\text{ChannelGate}(\mathbf{X})\right) \in \mathbb{R}^{B \times d \times H \times W}
$$

其中 $\text{ChannelGate}(\mathbf{X}) = \mathbf{X} \odot \sigma\!\left(\text{MLP}\!\left(\text{GAP}(\mathbf{X})\right)\right)$，$\text{Proj}_{1\times1}$ 为 Conv $1\times1$ + BN + SiLU。

**双路径拼接融合**：将两条路径的输出沿通道维度拼接后，经 $3\times3$ 卷积映射到 $K$ 个分支的路由 logits，再沿分支维度 Softmax 归一化：

$$
\mathbf{W} = \text{Softmax}_{k}\left(\text{Conv}_{3 \times 3}\left([\mathbf{F}_{\text{strip}};\; \mathbf{F}_{\text{ch}}]\right)\right) \in \mathbb{R}^{B \times K \times H \times W}
$$

其中 $[\cdot\,;\,\cdot]$ 表示通道维拼接，输入维度为 $2d$，输出维度为 $K=3$。$\mathbf{W}(b,k,h,w)$ 即空间位置 $(h,w)$ 对第 $k$ 个尺度分支的亲和力权重。

> **设计动机**：条带池化擅长编码「目标在空间哪个位置、沿什么方向」，但对「当前位置由哪些通道主导」不敏感。通道门控路径补充了全局通道选择性信息，两者互补，使尺度预测同时感知空间结构和通道语义（Channel-Spatial Joint Routing）。

**SADR 完整结构示意图**：

```
输入特征 X (B, C, H, W)
    │
    ├─────────────┬──────────────┬────────────────────────────────────┐
    ▼             ▼              ▼                                    ▼
┌─────────┐ ┌─────────┐ ┌──────────┐      ┌─────── Scale Predictor (尺度预测网络) ───────┐
│Branch-S │ │Branch-M │ │Branch-L  │      │                                              │
│ k=3,d=1 │ │ k=3,d=3 │ │ k=3,d=5  │      │  路径A: 条带池化        路径B: 通道门控       │
│ 小感受野 │ │ 中感受野 │ │ 大感受野  │      │  StripPool_H           ChannelGate(SE)      │
│ (局部)   │ │ (区域)   │ │ (全局)    │      │  StripPool_V           → Proj 1×1Conv      │
│ +SE校准  │ │ +SE校准  │ │ +SE校准   │      │  → 1×1Conv → F_strip   → F_ch              │
└───┬─────┘ └───┬─────┘ └───┬──────┘      │                                              │
    │           │            │             │  [F_strip ; F_ch] (通道拼接, 2d维)            │
    │  F_s      │  F_m       │  F_l        │  → Conv 3×3 → Softmax(dim=k)                │
    │           │            │             │  → W (B, 3, H, W)                            │
    │           │            │             └───────────────┬──────────────────────────────┘
    │           │            │                             │
    │           │            │          W_s, W_m, W_l (逐像素尺度权重)
    │           │            │                             │
    └───────────┴────────────┴─────────────────────────────┘
                             │
                  ┌──────────▼───────────┐
                  │ 逐像素加权融合:       │
                  │ F_out(b,:,h,w) =     │
                  │   W_s·F_s + W_m·F_m  │
                  │   + W_l·F_l          │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │ 1×1 Conv + BN        │
                  │ + γ·残差连接 (γ₀=0)   │
                  └──────────┬───────────┘
                             │
                  输出 Y (B, C, H, W)
```

**Step 3：逐像素加权融合 + 残差**

$$
\mathbf{F}_{\text{fused}} = \sum_{k=1}^{K} \mathbf{W}_{:,k,:,:} \odot \mathbf{F}_k
$$

$$
\mathbf{Y} = \mathbf{X} + \gamma \cdot \text{Conv}_{1 \times 1}(\mathbf{F}_{\text{fused}})
$$

---

---

## 二、BDFR：背景解耦特征精炼模块 (Background-Decoupled Feature Refinement)

### 2.1 模块总览

BDFR 的核心思想是：**通过可学习的背景原型向量建模道路背景模式，计算每个空间位置与背景原型的"偏离度"，将偏离度高的区域（潜在异常目标）以注意力方式增强**。

该模块同样融合了多个经典方法的思想，并针对道路异常检测场景进行了创新组合。

---

### 2.2 组件出处与公式

#### 2.2.1 原型学习 (Prototype Learning) —— 出自 Prototypical Networks / 原型网络

**出处**：

> **Snell J, Swersky K, Zemel R.** Prototypical Networks for Few-shot Learning[C]. *Advances in Neural Information Processing Systems (NeurIPS)*, 2017: 4077-4087.

> **Wang Y, Xu C, Liu C, et al.** Instance Credibility Inference for Few-Shot Learning[C]. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020: 12836-12845.

**核心思想**：在嵌入空间中为每个类别维护一个"原型"向量（类中心），通过计算查询样本到各原型的距离来进行分类。

**原型定义**：

$$
\mathbf{p}_k = \frac{1}{|S_k|} \sum_{\mathbf{x}_i \in S_k} f_\theta(\mathbf{x}_i), \quad k = 1, \ldots, K
$$

其中 $S_k$ 为属于第 $k$ 个类别的样本集合，$f_\theta$ 为特征提取网络。

**本文改进**：
- 原始原型网络为每个**类别**维护一个原型，本文为**道路背景**维护 $K=8$ 个原型向量，对应不同的背景模式（如沥青路面、白色标线、绿化带、护栏等）；
- 原型不通过样本均值计算，而是作为**可学习参数** $\mathbf{P}_{\text{bg}} \in \mathbb{R}^{K \times D}$ 端到端优化；
- 使用 Xavier 均匀初始化：$\mathbf{P}_{\text{bg}} \sim \mathcal{U}\left(-\sqrt{6/(K+D)}, \sqrt{6/(K+D)}\right)$。

---

#### 2.2.2 特征投影层 (Feature Projection)

**出处**（$1 \times 1$ 卷积降维技术）：

> **Lin M, Chen Q, Yan S.** Network In Network[C]. *International Conference on Learning Representations (ICLR)*, 2014.

$1 \times 1$ 卷积作为跨通道的线性投影，最早由 NIN 提出用于增强网络表达能力，后被广泛用于特征降维以降低计算开销（如 GoogLeNet [Szegedy et al., 2015]、ResNet Bottleneck [He et al., 2016]）。

**公式**：

$$
\mathbf{Z} = \phi\left(\text{BN}\left(\text{DW-Conv}_{3 \times 3}\left(\phi\left(\text{BN}\left(\text{Conv}_{1 \times 1}(\mathbf{X})\right)\right)\right)\right)\right) \in \mathbb{R}^{B \times D \times H \times W}
$$

其中 $D = \max(C/4, 32)$ 为投影维度，将高维特征投影到低维空间以降低后续距离计算的开销。

---

#### 2.2.3 偏离度计算 (Deviation Computation) —— 受异常检测理论启发

**出处**（深度异常检测中的距离度量思想）：

> **Ruff L, Vandermeulen R, Goernitz N, et al.** Deep One-Class Classification[C]. *International Conference on Machine Learning (ICML)*, 2018: 4393-4402.

> **Defard T, Setkov A, Loesch A, et al.** PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization[C]. *International Conference on Pattern Recognition (ICPR)*, 2021: 475-489.

> **Roth K, Pemula L, Schiber J, et al.** Towards Total Recall in Industrial Anomaly Detection[C]. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022: 14318-14328.

**核心思想**：在异常检测领域，通过计算测试样本到"正常分布中心"的距离来度量异常程度。Deep SVDD 将所有正常样本映射到一个超球中心，PaDiM 使用多元高斯建模每个 patch 的正常分布。

**本文偏离度公式推导**：

设投影特征为 $\mathbf{Z} \in \mathbb{R}^{B \times D \times H \times W}$，reshape 为 $\hat{\mathbf{Z}} \in \mathbb{R}^{B \times N \times D}$（$N = H \times W$），背景原型为 $\mathbf{P} \in \mathbb{R}^{K \times D}$。

**Step 1：计算到所有原型的 $L_2$ 距离**

$$
d_{n,k} = \| \hat{\mathbf{z}}_n - \mathbf{p}_k \|_2, \quad n = 1, \ldots, N, \quad k = 1, \ldots, K
$$

即 $\mathbf{D} = \text{cdist}(\hat{\mathbf{Z}}, \mathbf{P}) \in \mathbb{R}^{B \times N \times K}$。

**Step 2：取到最近原型的最小距离**

$$
d_n^{\min} = \min_{k=1}^{K} d_{n,k}
$$

直觉：如果一个空间位置的特征与所有背景原型都很远，说明它很可能是前景异常区域。

**Step 3：温度缩放 + Min-Max 归一化**

$$
\tilde{d}_n = \frac{d_n^{\min}}{\tau + \epsilon}
$$

$$
\delta_n = \frac{\tilde{d}_n - \min(\tilde{\mathbf{d}})}{\max(\tilde{\mathbf{d}}) - \min(\tilde{\mathbf{d}}) + \epsilon}
$$

其中 $\tau$ 为温度参数（控制偏离度映射的对比度），$\epsilon = 10^{-6}$ 防止除零。

最终得到偏离度图 $\boldsymbol{\delta} \in \mathbb{R}^{B \times 1 \times H \times W}$，值域 $[0, 1]$。

**温度参数 $\tau$ 的作用**：

温度缩放是一种常见的距离/相似度调节技巧，在对比学习（如 SimCLR [Chen et al., 2020]）和知识蒸馏（如 Hinton et al., 2015）中均有广泛应用。本文借鉴其思想，将 $\tau$ 作为线性缩放因子调节偏离度映射的对比度：$\tau$ 越小，映射越"尖锐"（前景-背景分界更明显）；$\tau$ 越大，映射越"平滑"。本文默认 $\tau = 1.0$。

> 注：此处 $\tau$ 为简单的线性缩放而非 Softmax 温度，与 SimCLR 中的用法有所不同，仅借鉴其"通过标量因子控制分布锐度"的一般思想。

---

#### 2.2.4 EMA 原型更新 (Exponential Moving Average) —— 出自 Mean Teacher / MoCo

**出处**：

> **Tarvainen A, Valpola H.** Mean Teachers are Better Role Models: Weight-averaged Consistency Targets Improve Semi-supervised Learning Results[C]. *Advances in Neural Information Processing Systems (NeurIPS)*, 2017: 1195-1204.

> **He K, Fan H, Wu Y, et al.** Momentum Contrast for Unsupervised Visual Representation Learning[C]. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020: 9729-9738.

**核心思想**：使用指数移动平均 (EMA) 更新参数，使目标分布平滑变化，防止训练震荡。

**EMA 更新公式**：

$$
\bar{\mathbf{p}}_k^{(t)} = m \cdot \bar{\mathbf{p}}_k^{(t-1)} + (1 - m) \cdot \hat{\mathbf{p}}_k^{(t)}
$$

其中 $m = 0.996$ 为动量系数，$\hat{\mathbf{p}}_k^{(t)}$ 为当前 batch 中分配到第 $k$ 个原型的特征均值。

**原型分配策略**（使用余弦相似度，而非 $L_2$ 距离，提高稳定性）：

$$
a_n = \arg\max_{k} \frac{\hat{\mathbf{z}}_n \cdot \mathbf{p}_k}{\|\hat{\mathbf{z}}_n\| \cdot \|\mathbf{p}_k\|}
$$

$$
\hat{\mathbf{p}}_k^{(t)} = \frac{1}{|\{n : a_n = k\}|} \sum_{n : a_n = k} \hat{\mathbf{z}}_n, \quad \text{当} \ |\{n : a_n = k\}| > 10
$$

**软同步回可学习参数**（防止 EMA 和梯度更新方向分歧过大）：

$$
\mathbf{P}_{\text{bg}}^{(t)} \leftarrow 0.95 \cdot \mathbf{P}_{\text{bg}}^{(t)} + 0.05 \cdot \bar{\mathbf{P}}_{\text{ema}}^{(t)}
$$

---

#### 2.2.5 偏离度→注意力映射 (Deviation-to-Attention Head)

**出处**（注意力机制的一般框架）：

> **Woo S, Park J, Lee J Y, et al.** CBAM: Convolutional Block Attention Module[C]. *European Conference on Computer Vision (ECCV)*, 2018: 3-19.

**公式**：

偏离度图 $\boldsymbol{\delta} \in \mathbb{R}^{B \times 1 \times H \times W}$ 经过双分支深度可分离卷积处理：

$$
\mathbf{A}_{\text{small}} = \text{DWSConv}_{3 \times 3}(\boldsymbol{\delta}), \quad \mathbf{A}_{\text{large}} = \text{DWSConv}_{5 \times 5}(\boldsymbol{\delta})
$$

$$
\mathbf{A} = \sigma\left(\text{Conv}_{1 \times 1}\left([\mathbf{A}_{\text{small}}; \mathbf{A}_{\text{large}}]\right)\right) \in \mathbb{R}^{B \times 1 \times H \times W}
$$

其中 $\sigma$ 为 Sigmoid 函数，$\mathbf{A}$ 值域 $[0, 1]$，表示每个空间位置的"异常响应强度"。

**双路径设计的目的**：$3 \times 3$ 路径捕获局部偏离模式（小目标如抛洒物），$5 \times 5$ 路径捕获区域偏离模式（大目标如违停车辆）。

---

#### 2.2.6 注意力加权 + 残差输出

**公式**：

$$
\mathbf{Y} = \mathbf{X} + \gamma \cdot \left( \mathbf{X} \odot \mathbf{A} \right)
$$

其中 $\odot$ 为逐元素乘法（Hadamard 积），$\mathbf{A} \in [0, 1]$ 为偏离度注意力图，$\gamma$ 为可学习标量（初始化为 $0$，LayerScale 技巧）。

**恒等映射验证**：当 $\gamma = 0$ 时：

$$
\mathbf{Y} = \mathbf{X} + 0 \cdot (\mathbf{X} \odot \mathbf{A}) = \mathbf{X}
$$

即训练初期模块为严格的恒等映射（Identity），不会破坏骨干网络的预训练特征流。随着训练推进，$\gamma$ 逐渐增大，模块学会利用偏离度注意力 $\mathbf{A}$ 对前景异常区域进行自适应增强。

**直觉理解**：$\mathbf{X} \odot \mathbf{A}$ 提取了"偏离背景的特征分量"，$\gamma$ 控制这部分增强的强度，最终与原始特征相加构成残差连接。

---

### 2.3 BDFR 完整前向传播公式汇总

设输入为 $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$：

**Step 1：特征投影**

$$
\mathbf{Z} = f_{\text{proj}}(\mathbf{X}) \in \mathbb{R}^{B \times D \times H \times W}, \quad D = \max(C/4, 32)
$$

**Step 2：EMA 原型更新**（仅训练时）

$$
\bar{\mathbf{P}}_k \leftarrow m \cdot \bar{\mathbf{P}}_k + (1 - m) \cdot \text{ClusterMean}_k(\mathbf{Z})
$$

**Step 3：偏离度计算**

$$
\boldsymbol{\delta}(b, 1, h, w) = \text{Normalize}\left( \frac{\min_k \|\mathbf{z}_{b,h,w} - \mathbf{p}_k\|_2}{\tau} \right) \in [0, 1]
$$

**Step 4：偏离度→注意力**

$$
\mathbf{A} = \sigma\left(\text{DualBranchConv}(\boldsymbol{\delta})\right) \in [0, 1]
$$

**Step 5：注意力加权 + 残差**

$$
\mathbf{Y} = \mathbf{X} + \gamma \cdot (\mathbf{X} \odot \mathbf{A})
$$

当 $\gamma = 0$ 时 $\mathbf{Y} = \mathbf{X}$（严格恒等映射）。

---

---

## 三、综合参考文献列表

以下为 SADR 和 BDFR 涉及的全部核心参考文献：

### 条带池化
1. **Hou Q, Zhang L, Cheng M M, et al.** Strip Pooling: Rethinking Spatial Pooling for Scene Parsing[C]. CVPR, 2020: 4003-4012.

### 膨胀卷积 / 空洞空间金字塔池化
2. **Chen L C, Papandreou G, Kokkinos I, et al.** DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs[J]. TPAMI, 2018, 40(4): 834-848.
3. **Chen L C, Zhu Y, Papandreou G, et al.** Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation[C]. ECCV, 2018: 801-818.

### 通道注意力 / SE 模块
4. **Hu J, Shen L, Sun G.** Squeeze-and-Excitation Networks[C]. CVPR, 2018: 7132-7141.

### 动态路由 / 核选择
5. **Li X, Wang W, Hu X, et al.** Selective Kernel Networks[C]. CVPR, 2019: 510-519.
6. **Yang B, Bender G, Le Q V, et al.** CondConv: Conditionally Parameterized Convolutions for Efficient Inference[C]. NeurIPS, 2019: 1307-1318.
7. **Chen Y, Dai X, Liu M, et al.** Dynamic Convolution: Attention over Convolution Kernels[C]. CVPR, 2020: 11030-11039.

### 深度可分离卷积
8. **Howard A G, Zhu M, Chen B, et al.** MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications[J]. arXiv:1704.04861, 2017.

### 可学习残差缩放 / LayerScale
9. **Touvron H, Cord M, Sablayrolles A, et al.** Going Deeper with Image Transformers[C]. ICCV, 2021: 32-42.

### 原型学习
10. **Snell J, Swersky K, Zemel R.** Prototypical Networks for Few-shot Learning[C]. NeurIPS, 2017: 4077-4087.

### EMA 动量更新
11. **Tarvainen A, Valpola H.** Mean Teachers are Better Role Models: Weight-averaged Consistency Targets Improve Semi-supervised Learning Results[C]. NeurIPS, 2017: 1195-1204.
12. **He K, Fan H, Wu Y, et al.** Momentum Contrast for Unsupervised Visual Representation Learning[C]. CVPR, 2020: 9729-9738.

### 深度异常检测 / 偏离度度量
13. **Ruff L, Vandermeulen R, Goernitz N, et al.** Deep One-Class Classification[C]. ICML, 2018: 4393-4402.
14. **Defard T, Setkov A, Loesch A, et al.** PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization[C]. ICPR, 2021: 475-489.
15. **Roth K, Pemula L, Schiber J, et al.** Towards Total Recall in Industrial Anomaly Detection[C]. CVPR, 2022: 14318-14328.

### 对比学习 / 温度缩放（思想借鉴）
16. **Chen T, Kornblith S, Norouzi M, et al.** A Simple Framework for Contrastive Learning of Visual Representations[C]. ICML, 2020: 1597-1607.

### 注意力机制
17. **Woo S, Park J, Lee J Y, et al.** CBAM: Convolutional Block Attention Module[C]. ECCV, 2018: 3-19.

### Network In Network / 1×1 卷积
18. **Lin M, Chen Q, Yan S.** Network In Network[C]. ICLR, 2014.

### YOLO 基线
19. **Jocher G, Chaurasia A, Qiu J.** Ultralytics YOLO[EB/OL]. 2024. https://docs.ultralytics.com

> ⚠️ **YOLO 引用说明**：Ultralytics YOLO 系列（含 YOLOv5/v8/v11）目前无正式学术论文发表。在正式论文中建议采用以下方式之一：
> - 引用 Ultralytics 官方文档或 GitHub 仓库作为技术报告；
> - 引用 YOLOv5 的 Zenodo DOI：Jocher G. ultralytics/yolov5: v7.0[Z]. Zenodo, 2022. DOI:10.5281/zenodo.7347926
> - 或补充引用 YOLO 原始论文：Redmon J, et al. You Only Look Once: Unified, Real-Time Object Detection[C]. CVPR, 2016.
> - 具体版本号 "YOLO11" 为产品命名，论文中宜写为 "Ultralytics YOLO (v11)" 或 "YOLO11 (Ultralytics, 2024)"。

---

## 四、创新性总结

| 模块 | 核心创新 | 与现有方法的本质区别 |
|------|---------|-------------------|
| **SADR** | 逐像素尺度路由 + 条带池化驱动 | SKNet/CondConv 只在通道级或全图级做选择，SADR 在**每个像素位置**独立选择最优尺度组合；使用条带池化引入道路场景方向性先验 |
| **BDFR** | 可学习背景原型 + 偏离度驱动注意力 | 传统注意力 (SE/CBAM) 基于统计量增强显著特征，BDFR 基于**与背景原型的距离**增强偏离背景的区域；原型通过 EMA 稳定更新，端到端学习 |

> **论文撰写建议**：在论文中，SADR 和 BDFR 应定位为**创新的模块设计**（Novel Module Design），其创新性在于：(1) 将多个成熟技术进行了**面向道路异常检测的创新组合**；(2) 提出了现有方法不具备的**逐像素尺度路由**和**偏离度驱动注意力**两个新机制；(3) 通过消融实验验证各组件的有效性。每个子组件可在"相关工作"或"方法"章节中引用对应的原始论文以体现学术规范。
