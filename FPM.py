import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from models.sam2.build_sam import build_sam2
from vmamba import SS2D  # 2D Mamba 实现


# ==================== 基础 DoubleConv（用于解码器） ====================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# ==================== 全局随机性控制 ====================

def set_deterministic(seed: int = 42, use_deterministic_algorithms: bool = True):
    """
    统一设置随机种子和确定性开关，尽可能提升复现性。
    在调用训练/推理前执行一次。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if use_deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


# ==================== Adapter（SAM2 encoder 轻量微调） ====================

class Adapter(nn.Module):
    """
    将 SAM2 的每个 transformer block 包一层 Adapter：
      y = Block(x + MLP(x))
    冻结 Block 参数，仅训练 MLP 部分。
    """

    def __init__(self, blk):
        super(Adapter, self).__init__()
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU(),
        )
        self.block = blk

    def forward(self, x):
        return self.block(x + self.prompt_learn(x))


# ==================== 真实 3-band 频域块 StrongFreqBlock ====================

class StrongFreqBlock(nn.Module):
    """
    对输入特征 x 做真实 3-band 频域分解：

      x --FFT2+fftshift--> X
          |
          +-- radial masks --> X_low, X_mid, X_high
          |
         IFFT2 --> x_low0, x_mid0, x_high0 (三个空间域特征)

      然后做一个频带级 gate：
          GAP(x_low0, x_mid0, x_high0) -> softmax -> alpha_low/mid/high
          x_low  = alpha_low  * x_low0
          x_mid  = alpha_mid  * x_mid0
          x_high = alpha_high * x_high0

      最后 concat 三段，做一层卷积得到 fused 频域增强特征。

    输出:
        x_low, x_mid, x_high, x_fused    (四个都是与输入同尺寸同通道数)
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        r_low: float = 0.12,
        r_mid: float = 0.35,
        record_attn: bool = True,
        num_bases: int = 1,
    ):
        super().__init__()
        self.channels = channels
        # 可学习半径参数：通过 sigmoid 保持在 (0,1)，并在前向中强制 r_low < r_mid
        self.r_low_param = nn.Parameter(torch.tensor(float(r_low)))
        self.r_mid_param = nn.Parameter(torch.tensor(float(r_mid)))
        self.record_attn = record_attn
        self.num_bases = num_bases

        hidden = max(channels // reduction, 8)

        # 频带级 gate：基于三个 band 的 GAP，输出 low/mid/high 的权重
        self.band_mlp = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=True),  # [B,3,1,1]
        )

        # 单个径向高斯基，控制带宽/中心
        if num_bases > 0:
            self.basis_mu = nn.Parameter(torch.tensor([0.5] * num_bases, dtype=torch.float32))
            self.basis_sigma = nn.Parameter(torch.tensor([0.15] * num_bases, dtype=torch.float32))
            self.basis_proj = nn.Conv2d(num_bases * channels, channels, kernel_size=1, bias=False)
        else:
            self.basis_mu = None
            self.basis_sigma = None
            self.basis_proj = None

        # 三段 concat + 基后再做一层卷积融合
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels * 3 + (channels if num_bases > 0 else 0), channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # 可视化缓存
        self.last_mag = None        # [H,W] 频谱幅度（log）
        self.last_alpha = None      # [3]    三个频带的平均权重

    @staticmethod
    def _make_radial_masks(H, W, r_low, r_mid, device):
        """
        生成三个半径掩码：low / mid / high
        半径归一化到 [0,1]，r_low, r_mid 分别是两个分割阈值。
        """
        ys = torch.arange(H, device=device).float() - (H - 1) / 2.0
        xs = torch.arange(W, device=device).float() - (W - 1) / 2.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        rr = torch.sqrt(yy ** 2 + xx ** 2)
        rr = rr / (rr.max() + 1e-8)

        mask_low = (rr <= r_low).float()
        mask_mid = ((rr > r_low) & (rr <= r_mid)).float()
        mask_high = (rr > r_mid).float()

        # [1,1,H,W]
        mask_low = mask_low.unsqueeze(0).unsqueeze(0)
        mask_mid = mask_mid.unsqueeze(0).unsqueeze(0)
        mask_high = mask_high.unsqueeze(0).unsqueeze(0)
        return mask_low, mask_mid, mask_high

    def forward(self, x):
        """
        x: [B,C,H,W]
        return: x_low, x_mid, x_high, x_fused
        """
        dtype = x.dtype
        B, C, H, W = x.shape
        device = x.device

        # ---- 1. FFT2 + fftshift ----
        x_float = x.to(torch.float32)
        X = torch.fft.fft2(x_float, norm="ortho")             # [B,C,H,W] complex64
        X_shift = torch.fft.fftshift(X, dim=(-2, -1))

        # ---- 2. 半径掩码划分三段 ----
        # 将可学习参数映射到合法半径，并保证 r_low < r_mid
        r_low = torch.sigmoid(self.r_low_param)
        # 通过累乘约束 r_mid 始终大于 r_low
        r_mid = r_low + torch.sigmoid(self.r_mid_param) * (1.0 - r_low)

        m_low, m_mid, m_high = self._make_radial_masks(
            H, W, r_low, r_mid, device
        )
        X_low_s = X_shift * m_low
        X_mid_s = X_shift * m_mid
        X_high_s = X_shift * m_high

        # 移回左上角，做 IFFT2
        X_low = torch.fft.ifftshift(X_low_s, dim=(-2, -1))
        X_mid = torch.fft.ifftshift(X_mid_s, dim=(-2, -1))
        X_high = torch.fft.ifftshift(X_high_s, dim=(-2, -1))

        x_low0 = torch.fft.ifft2(X_low, norm="ortho").real   # [B,C,H,W]
        x_mid0 = torch.fft.ifft2(X_mid, norm="ortho").real
        x_high0 = torch.fft.ifft2(X_high, norm="ortho").real

        # ---- 3. 频带级 gate ----
        g_low = F.adaptive_avg_pool2d(x_low0, 1)   # [B,C,1,1]
        g_mid = F.adaptive_avg_pool2d(x_mid0, 1)
        g_high = F.adaptive_avg_pool2d(x_high0, 1)

        g_cat = torch.cat([g_low, g_mid, g_high], dim=1)  # [B,3C,1,1]
        logits = self.band_mlp(g_cat)                    # [B,3,1,1]
        alpha = torch.softmax(logits, dim=1)             # [B,3,1,1]

        alpha_low = alpha[:, 0:1]
        alpha_mid = alpha[:, 1:2]
        alpha_high = alpha[:, 2:3]

        x_low = alpha_low * x_low0
        x_mid = alpha_mid * x_mid0
        x_high = alpha_high * x_high0

        # ---- 4. 径向基（可选） + 融合 ----
        basis_feat = None
        if self.num_bases and self.num_bases > 0:
            ys = torch.arange(H, device=device).float() - (H - 1) / 2.0
            xs = torch.arange(W, device=device).float() - (W - 1) / 2.0
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            rr = torch.sqrt(yy ** 2 + xx ** 2)
            rr = rr / (rr.max() + 1e-8)

            basis_list = []
            for mu, sigma in zip(self.basis_mu, self.basis_sigma):
                mask = torch.exp(-0.5 * ((rr - mu) / (sigma.abs() + 1e-3)) ** 2)
                mask = mask.to(device=device, dtype=x_float.dtype).unsqueeze(0).unsqueeze(0)
                X_b = X_shift * mask
                X_b = torch.fft.ifftshift(X_b, dim=(-2, -1))
                x_b = torch.fft.ifft2(X_b, norm="ortho").real
                basis_list.append(x_b)
            basis_feat = torch.cat(basis_list, dim=1)  # [B,num_bases*C,H,W]
            basis_feat = self.basis_proj(basis_feat)   # [B,C,H,W]
        else:
            basis_feat = torch.zeros_like(x_low)

        x_cat = torch.cat([x_low, x_mid, x_high, basis_feat], dim=1)     # [B,3C(+C),H,W]
        x_fused = self.fuse_conv(x_cat)                      # [B,C,H,W]

        # 回到原 dtype
        x_low = x_low.to(dtype)
        x_mid = x_mid.to(dtype)
        x_high = x_high.to(dtype)
        x_fused = x_fused.to(dtype)

        # ---- 5. 可视化记录 ----
        if self.record_attn:
            with torch.no_grad():
                mag = torch.abs(X_shift[0, 0])
                mag = torch.log1p(mag)
                self.last_mag = mag.detach().cpu()           # [H,W]

                alpha_mean = alpha.mean(dim=(0, 2, 3))       # [3]
                self.last_alpha = alpha_mean.detach().cpu()

        return x_low, x_mid, x_high, x_fused


# ==================== 输入图像 3-band 频域金字塔 ====================

class InputFreqEncoder(nn.Module):
    """
    从原始输入图像提取多尺度 + 三频带频域先验：

      x -> stem -> (freq1, down) -> (freq2, down) -> (freq3, down) -> freq4

    每个尺度 l 输出:
      (F_low^l, F_mid^l, F_high^l) ∈ R^{B×C_out×H_l×W_l}

    用法：
      priors = self.freq_encoder(x)
      f1_low, f1_mid, f1_high = priors["s1"]
      ...
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        out_channels=64,
        reduction=4,
        r_low=0.12,
        r_mid=0.35,
        record_attn: bool = True,
    ):
        super().__init__()

        self.stem = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1, bias=False
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.freq1 = StrongFreqBlock(c1, reduction=reduction,
                                     r_low=r_low, r_mid=r_mid,
                                     record_attn=record_attn)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)

        self.freq2 = StrongFreqBlock(c2, reduction=reduction,
                                     r_low=r_low, r_mid=r_mid,
                                     record_attn=record_attn)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)

        self.freq3 = StrongFreqBlock(c3, reduction=reduction,
                                     r_low=r_low, r_mid=r_mid,
                                     record_attn=record_attn)
        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)

        self.freq4 = StrongFreqBlock(c4, reduction=reduction,
                                     r_low=r_low, r_mid=r_mid,
                                     record_attn=record_attn)

        # 每个尺度一个 1×1 Conv，将 band 特征统一到 out_channels 维
        self.proj1 = nn.Conv2d(c1, out_channels, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(c2, out_channels, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(c3, out_channels, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(c4, out_channels, kernel_size=1, bias=False)

    @staticmethod
    def _proj_triplet(proj, low, mid, high):
        """同一 1×1 conv 分别作用于 low/mid/high。"""
        return proj(low), proj(mid), proj(high)

    def forward(self, x):
        # x: [B,3,H,W]
        x = self.stem(x)  # [B,c1,H,W]

        f1_low, f1_mid, f1_high, f1_fused = self.freq1(x)
        x = self.down1(f1_fused)  # [B,c2,H/2,W/2]

        f2_low, f2_mid, f2_high, f2_fused = self.freq2(x)
        x = self.down2(f2_fused)  # [B,c3,H/4,W/4]

        f3_low, f3_mid, f3_high, f3_fused = self.freq3(x)
        x = self.down3(f3_fused)  # [B,c4,H/8,W/8]

        f4_low, f4_mid, f4_high, _ = self.freq4(x)

        # 投影到统一通道维
        f1_low, f1_mid, f1_high = self._proj_triplet(self.proj1, f1_low, f1_mid, f1_high)
        f2_low, f2_mid, f2_high = self._proj_triplet(self.proj2, f2_low, f2_mid, f2_high)
        f3_low, f3_mid, f3_high = self._proj_triplet(self.proj3, f3_low, f3_mid, f3_high)
        f4_low, f4_mid, f4_high = self._proj_triplet(self.proj4, f4_low, f4_mid, f4_high)

        priors = {
            "s1": (f1_low, f1_mid, f1_high),
            "s2": (f2_low, f2_mid, f2_high),
            "s3": (f3_low, f3_mid, f3_high),
            "s4": (f4_low, f4_mid, f4_high),
        }
        return priors


# ==================== 占位 InformationFilter（保持旧接口兼容） ====================

class InformationFilter(nn.Module):
    """
    兼容旧训练脚本：
      - 现在不再使用 IFM，把它当作恒等映射。
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x_sem, x_mamba, x_freq):
        return x_mamba


# ==================== Conv + Mamba + 频域先验（两级 gate）解码块 ====================

class MambaPriorUp(nn.Module):
    """
    单层解码模块（增强版）：

      输入：
        dec_feat     : 上一层解码特征（或最深层 encode 特征） [B,Cd,Hd,Wd]
        skip_feat    : 对应尺度的 encoder skip 特征       [B,Cs,He,We]
        prior_low    : 该尺度频域先验 - 低频             [B,Cp,Hp,Wp]
        prior_mid    : 该尺度频域先验 - 中频
        prior_high   : 该尺度频域先验 - 高频

      结构：
        1) dec_feat 上采样 + 与 skip 拼接 → DoubleConv 得到 h_sem
        2) h_sem → SS2D Mamba → h_mamba
        3) 频带级 α_l (global)：基于 GAP(h_mamba, priors) → softmax
        4) 空间级 β_l(x)：基于 [h_mamba, prior_mix]  → β ∈ [0,1]^{B×1×H×W}
           prior_mix = α_low * F_low + α_mid * F_mid + α_high * F_high
        5) 高频残差支路（可选）：
             edge_raw  = prior_high - prior_low   （高通近似）
             edge_feat = Conv1x1(edge_raw)
             edge_gate = sigmoid(Conv([h_mamba, edge_feat]))
             out = h_mamba + β ⊙ prior_mix + edge_gate ⊙ edge_feat
        6) 语义相位调制（SPM）：基于 h_mamba 生成 Δφ，注入中/高频相位，增强边界/结构
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_state: int = 16,
        record_attn: bool = True,
        high_freq_bias: float = 0.0,   # 对 high band logit 的偏置，用于鼓励高频（不同层可设置不同值）
        use_edge_refine: bool = False,  # 默认关闭高频残差细化，保持精简
        prior_rank: int | None = None,
        residual_scale: float = 0.5,
        max_phase_shift: float = 0.523599,  # ≈π/6，SPM 相位偏移上限
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

        # Mamba 语义分支
        self.mamba = SS2D(d_model=out_channels, d_state=d_state)
        self.prior_rank = prior_rank or max(out_channels // 4, 8)
        self.prior_proj = nn.Conv2d(out_channels, self.prior_rank, kernel_size=1, bias=False)
        self.prior_back = nn.Conv2d(self.prior_rank, out_channels, kernel_size=1, bias=False)
        self.residual_scale = residual_scale


        # 频带级路由 α_l（global）
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.alpha_mlp = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 3, kernel_size=1, bias=True),
        )

        # 空间级 gate β_l(x)
        mid_channels = max(out_channels // 2, 8)
        self.beta_conv = nn.Sequential(
            nn.Conv2d(
                out_channels + self.prior_rank, mid_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        # 高频残差细化分支（基于 prior_high - prior_low）
        self.use_edge_refine = use_edge_refine
        if use_edge_refine:
            self.edge_proj = nn.Conv2d(
                out_channels, out_channels, kernel_size=1, bias=False
            )
            edge_mid = max(out_channels // 2, 8)
            self.edge_gate = nn.Sequential(
                nn.Conv2d(
                    out_channels * 2,
                    edge_mid,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(edge_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(edge_mid, out_channels, kernel_size=3, padding=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.edge_proj = None
            self.edge_gate = None

        # 记录可视化信息
        self.record_attn = record_attn
        self.high_freq_bias = high_freq_bias
        self.max_phase_shift = max_phase_shift

        self.last_alpha = None       # [B,3]
        self.last_beta = None        # [B,1,H,W]
        self.last_prior_low = None   # [B,C,H,W]
        self.last_prior_mid = None
        self.last_prior_high = None
        self.last_prior_mix = None   # [B,C,H,W]
        self.last_edge_gate = None   # [B,C,H,W] （可选）
        self.last_edge_feat = None   # [B,C,H,W] （可选）

    def forward(self, dec_feat, skip_feat, prior_low, prior_mid, prior_high):
        # 1) 上采样 + 尺度对齐
        x1 = self.up(dec_feat)
        diffY = skip_feat.size()[2] - x1.size()[2]
        diffX = skip_feat.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
            ],
        )

        # 2) concat 跳连 + DoubleConv 得到基础语义特征
        x = torch.cat([skip_feat, x1], dim=1)  # [B, in_channels, H, W]
        h_sem = self.conv(x)                   # [B, C, H, W]
        B, C, H, W = h_sem.shape

        # 3) Mamba 语义分支
        x_bhwc = h_sem.permute(0, 2, 3, 1)  # [B,H,W,C]
        y_bhwc = self.mamba(x_bhwc)         # [B,H,W,C]
        h_mamba = y_bhwc.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]

        # 4) 对齐频域先验到当前分辨率
        if prior_low is None:
            prior_low = torch.zeros_like(h_mamba)
            prior_mid = torch.zeros_like(h_mamba)
            prior_high = torch.zeros_like(h_mamba)
        else:
            prior_low = F.interpolate(
                prior_low, size=(H, W), mode="bilinear", align_corners=False
            )
            prior_mid = F.interpolate(
                prior_mid, size=(H, W), mode="bilinear", align_corners=False
            )
            prior_high = F.interpolate(
                prior_high, size=(H, W), mode="bilinear", align_corners=False
            )

        # 4.1) 语义相位调制：仅对中/高频注入 Δφ（受 h_mamba 驱动），低频保持不变
        prior_low, prior_mid, prior_high = self._semantic_phase_modulate(h_mamba, prior_low, prior_mid, prior_high)

        # 5) 频带级路由 α_l (global)
        g_m = self.gap(h_mamba)    # [B,C,1,1]
        g_l = self.gap(prior_low)
        g_mi = self.gap(prior_mid)
        g_h = self.gap(prior_high)

        g_cat = torch.cat([g_m, g_l, g_mi, g_h], dim=1)  # [B,4C,1,1]
        alpha_logits = self.alpha_mlp(g_cat).view(B, 3)  # [B,3]

        # 对 high band 加一点偏置（只对设置了 high_freq_bias 的层生效）
        if self.high_freq_bias != 0.0:
            alpha_logits[:, 2] = alpha_logits[:, 2] + self.high_freq_bias

        alpha = torch.softmax(alpha_logits, dim=1)       # [B,3]

        alpha_low = alpha[:, 0].view(B, 1, 1, 1)
        alpha_mid = alpha[:, 1].view(B, 1, 1, 1)
        alpha_high = alpha[:, 2].view(B, 1, 1, 1)

        prior_mix = (
            alpha_low * prior_low
            + alpha_mid * prior_mid
            + alpha_high * prior_high
        )  # [B,C,H,W]

        # 6) 低秩压缩 + 残差回投（缩放）
        prior_mix_proj = self.prior_proj(prior_mix)       # [B,prior_rank,H,W]
        prior_back = self.prior_back(prior_mix_proj) * self.residual_scale

        # 轻量 FiLM：使用 prior 的全局统计生成近恒等的缩放/偏置
        stats = self.gap(prior_mix_proj)
        gamma_beta = torch.zeros(
            (B, h_mamba.shape[1] * 2, 1, 1),
            device=prior_mix_proj.device,
            dtype=prior_mix_proj.dtype,
        )
        gamma, beta_film = gamma_beta.chunk(2, dim=1)
        h_dyn = h_mamba * (1.0 + gamma) + beta_film

        # 7) 空间级 gate β_l(x)
        gate_in = torch.cat([h_dyn, prior_mix_proj], dim=1)  # [B,C+prior_rank,H,W]
        beta = self.beta_conv(gate_in)                    # [B,1,H,W]

        out = h_dyn + beta * (prior_mix + prior_back)     # [B,C,H,W]

        # 7) 高频残差细化分支（可选）
        edge_gate = None
        edge_feat = None
        if self.use_edge_refine and self.edge_proj is not None:
            # 简单高通：高频 - 低频
            edge_raw = prior_high - prior_low             # [B,C,H,W]
            edge_feat = self.edge_proj(edge_raw)         # [B,C,H,W]

            edge_in = torch.cat([h_mamba, edge_feat], dim=1)  # [B,2C,H,W]
            edge_gate = self.edge_gate(edge_in)               # [B,C,H,W]

            out = out + edge_gate * edge_feat

        # 8) 可视化缓存
        if self.record_attn:
            with torch.no_grad():
                self.last_alpha = alpha.detach()
                self.last_beta = beta.detach()
                self.last_prior_low = prior_low.detach()
                self.last_prior_mid = prior_mid.detach()
                self.last_prior_high = prior_high.detach()
                self.last_prior_mix = prior_mix.detach()
                if edge_gate is not None:
                    self.last_edge_gate = edge_gate.detach()
                if edge_feat is not None:
                    self.last_edge_feat = edge_feat.detach()

        return out

    def _semantic_phase_modulate(self, h_sem, prior_low, prior_mid, prior_high):
        """
        语义驱动的相位注入（SPM）：对中/高频频段注入 Δφ，低频不变。
        Δφ 由 h_sem（Mamba 语义特征）生成，限制在 [-max_phase_shift, max_phase_shift]。
        """
        B, C, H, W = h_sem.shape
        device = h_sem.device
        dtype = h_sem.dtype
        # 生成相位偏移场
        delta_phi = torch.tanh(F.adaptive_avg_pool2d(h_sem, (H, W)))
        delta_phi = delta_phi.mean(dim=1, keepdim=True)  # [B,1,H,W]
        delta_phi = delta_phi * self.max_phase_shift

        def modulate(prior_band):
            # FFT
            X = torch.fft.fft2(prior_band.to(torch.float32), norm="ortho")
            mag = torch.abs(X)
            phase = torch.angle(X)
            phase_mod = phase + delta_phi
            X_mod = mag * torch.exp(1j * phase_mod)
            x_mod = torch.fft.ifft2(X_mod, norm="ortho").real
            return x_mod.to(dtype)

        prior_mid_mod = modulate(prior_mid)
        prior_high_mod = modulate(prior_high)
        # 低频保持不变
        return prior_low, prior_mid_mod, prior_high_mod



# ==================== 主干模型：SAM2UNet（原图频域 → 先验） ====================

class SAM2UNet(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        base_channels: int = 32,
        dec_channels: int = 64,
        record_attn: bool = True,
    ):
        super().__init__()

        # ---- 构建 SAM2 编码器并裁掉用不到的部分 ----
        model_cfg = "sam2_hiera_l.yaml"
        model = build_sam2(model_cfg, checkpoint_path)
        print("SAM2 model built successfully.")

        # 去掉不需要的组件
        for attr in [
            "sam_mask_decoder",
            "sam_prompt_encoder",
            "memory_encoder",
            "memory_attention",
            "mask_downsample",
            "obj_ptr_tpos_proj",
            "obj_ptr_proj",
        ]:
            if hasattr(model, attr):
                delattr(model, attr)

        if hasattr(model.image_encoder, "neck"):
            del model.image_encoder.neck

        # Backbone：冻结 trunk 参数，只训练 Adapter + 解码器
        self.encoder = model.image_encoder.trunk
        for _, p in self.encoder.named_parameters():
            p.requires_grad = False
        self.encoder.blocks = nn.Sequential(
            *[Adapter(b) for b in self.encoder.blocks]
        )
        print("SAM2 encoder adapted with Adapters and trunk frozen.")

        # 对应 sam2_hiera_l 的 trunk 通道数: [144, 288, 576, 1152]
        self.conv_sam_x1 = nn.Conv2d(144, dec_channels, kernel_size=1)
        self.conv_sam_x2 = nn.Conv2d(288, dec_channels, kernel_size=1)
        self.conv_sam_x3 = nn.Conv2d(576, dec_channels, kernel_size=1)
        self.conv_sam_x4 = nn.Conv2d(1152, dec_channels, kernel_size=1)

        # 频域编码器：对原始输入图像做 4 级 3-band 频域先验
        self.freq_encoder = InputFreqEncoder(
            in_channels=3,
            base_channels=base_channels,
            out_channels=dec_channels,
            reduction=4,
            r_low=0.12,
            r_mid=0.35,
            record_attn=False,  # 默认关闭，可在评估时开启
        )

        # 解码器：3 层 Conv + Mamba + 频域先验
        # 最深层 up1：主要负责全局语义，基本不推高频
        self.up1 = MambaPriorUp(
            in_channels=dec_channels + dec_channels,
            out_channels=dec_channels,
            d_state=16,
            record_attn=record_attn,
            high_freq_bias=0.0,  # 几乎不偏向 high
            use_edge_refine=True,  # 也可以关掉看 ablation
        )

        # 中间层 up2：适度增加 high 的权重
        self.up2 = MambaPriorUp(
            in_channels=dec_channels + dec_channels,
            out_channels=dec_channels,
            d_state=16,
            record_attn=record_attn,
            high_freq_bias=0.15,  # 轻微偏向 high
            use_edge_refine=True,
        )

        # 最浅层 up3：最关心边界，明显偏高频
        self.up3 = MambaPriorUp(
            in_channels=dec_channels + dec_channels,
            out_channels=dec_channels,
            d_state=16,
            record_attn=record_attn,
            high_freq_bias=0.3,  # 明显鼓励 high band
            use_edge_refine=True,
        )

        # 三路输出（深监督）
        self.side3 = nn.Conv2d(dec_channels, 1, 1)  # up1
        self.side2 = nn.Conv2d(dec_channels, 1, 1)  # up2
        self.head = nn.Conv2d(dec_channels, 1, 1)   # up3

    def forward(self, x):
        """
        x: [B,3,H,W]
        返回: 主输出 + 两个深监督 (out1, out2, out3)，都统一到输入分辨率
        """
        B, _, H, W = x.shape

        # 1) 原图频域 3-band 金字塔先验
        freq_pyr = self.freq_encoder(x)
        f1_low, f1_mid, f1_high = freq_pyr["s1"]
        f2_low, f2_mid, f2_high = freq_pyr["s2"]
        f3_low, f3_mid, f3_high = freq_pyr["s3"]
        f4_low, f4_mid, f4_high = freq_pyr["s4"]

        # 2) SAM2 编码器多尺度特征（带 Adapter）
        s1, s2, s3, s4 = self.encoder(x)

        # 通道统一到 dec_channels
        x1_c = self.conv_sam_x1(s1)  # 最浅
        x2_c = self.conv_sam_x2(s2)
        x3_c = self.conv_sam_x3(s3)
        x4_c = self.conv_sam_x4(s4)  # 最深

        # 3) 解码：Conv + Mamba + 原图频域先验
        up1 = self.up1(x4_c, x3_c, f4_low, f4_mid, f4_high)  # 最深
        up2 = self.up2(up1, x2_c, f3_low, f3_mid, f3_high)
        up3 = self.up3(up2, x1_c, f2_low, f2_mid, f2_high)
        # s1 / f1_* 预留将来再加一层 up4

        # 4) 多尺度输出统一到输入大小
        out3 = F.interpolate(
            self.side3(up1), size=(H, W), mode="bilinear", align_corners=False
        )
        out2 = F.interpolate(
            self.side2(up2), size=(H, W), mode="bilinear", align_corners=False
        )
        out1 = F.interpolate(
            self.head(up3), size=(H, W), mode="bilinear", align_corners=False
        )

        return out1, out2, out3

    def get_ifm_reg_loss(self):
        """
        兼容旧训练脚本的接口：之前 IFM 有正则，这里直接返回 0。
        """
        return torch.tensor(
            0.0, device=self.head.weight.device, dtype=self.head.weight.dtype
        )


# ==================== 参数量统计 & 前向测试 ====================

def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam_ckpt_path = os.path.join(
        "/root/autodl-tmp/checkpoints/sam2.1_hiera_large.pt"
    )

    model = SAM2UNet(checkpoint_path=sam_ckpt_path).to(device)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params (M): {total_params / 1e6:.3f} M")
    print(f"Trainable params (M): {trainable_params / 1e6:.3f} M")

    x = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        out, out1, out2 = model(x)
        print("\n--- 模型前向传播成功 ---")
        print(f"Main output shape: {out.shape}")
        print(f"Side output 1 shape: {out1.shape}")
        print(f"Side output 2 shape: {out2.shape}")
