# mamba + StrongFreq (3-band) 可视化版训练脚本
import os
import argparse
import random
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings
import torch.nn as nn
import matplotlib.pyplot as plt  # 用于可视化

# 忽略特定警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_wavelets")

# 添加 models 目录到 Python 路径，确保可以导入 SAM2UNet
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')
sys.path.insert(0, models_dir)

# 从 models 目录导入 SAM2UNet 模型及 StrongFreqBlock（用于注册 hook）
from model9 import SAM2UNet, StrongFreqBlock, InformationFilter
# 从 dataset.py 导入数据集类
from dataset import FullDataset

# 导入自定义的工具函数
from utils import (
    brain_tumor_loss,
    compute_metrics,
    validate,
    plot_training_curve,
    apply_augmentation,
    check_nan_inf,
    val_collate_fn
)

# 检查是否有 amp 支持
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("⚠️ 混合精度训练不可用，将使用标准精度")


# ==================== 可视化频域 & 频带门控的辅助函数 ====================
# ==================== 可视化频域 & 频带门控的辅助函数 ====================
# ==================== 可视化频域 & MambaPriorUp 的辅助函数 ====================
def _radial_profile(mag_tensor: torch.Tensor):
    """
    计算 2D 频谱的径向平均分布，用于画 1D 频谱曲线。
    mag_tensor: [H, W] (在调用前应 .cpu())
    return: numpy array, shape [R_max+1]
    """
    mag = mag_tensor.numpy()
    H, W = mag.shape

    y, x = np.indices((H, W))
    center_y = (H - 1) / 2.0
    center_x = (W - 1) / 2.0
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    r = r.astype(np.int64)

    r_max = r.max()
    radial_sum = np.bincount(r.ravel(), weights=mag.ravel(), minlength=r_max + 1)
    radial_cnt = np.bincount(r.ravel(), minlength=r_max + 1)
    radial_mean = radial_sum / (radial_cnt + 1e-8)
    return radial_mean


def visualize_internal(model, val_loader, device, save_root, epoch, max_batches=1):
    """
    适配当前 SAM2UNet (model3.py) 的内部可视化：

    1) 频域编码器 InputFreqEncoder 内每个 StrongFreqBlock：
        - 频谱热力图 + 径向频率分布
        - 三频带 gate 柱状图 (low/mid/high)
        - 三个频带先验 map（low/mid/high，通道平均）

    2) 三个解码块 MambaPriorUp (up1/up2/up3)：
        - 路由权重 alpha (low/mid/high)
        - 空间 gate beta(x)
        - prior_mix 热力图
        - prior_low / prior_mid / prior_high 热力图
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # 全局风格设置
    plt.style.use("seaborn-v0_8")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.figsize": (6, 5),
        "axes.grid": False,
    })

    model.eval()

    # 取一个 batch 做分析
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        print("⚠️ 验证集为空，无法做内部可视化")
        return

    x = batch["image"].to(device, non_blocking=True)
    x = x[:1]  # 只看一张图，省显存

    # 目标目录
    analysis_dir = os.path.join(save_root, "analysis_v2", f"epoch_{epoch:03d}")
    os.makedirs(analysis_dir, exist_ok=True)

    # 用于判断到底有没有生成任何图
    made_any_figure = False

    # ------------------------------------------------------------------
    # 1) 注册 StrongFreqBlock 的 hook：记录 FFT 幅度、radial profile、三带先验 map
    # ------------------------------------------------------------------
    debug_state = {
        "freq_mag": [],    # (layer_name, mag[H,W], radial_profile[R])
        "freq_gate": [],   # (layer_name, probs[3])
        "freq_prior": [],  # (layer_name, band_name, prior_map[H,W])
    }
    handles = []

    for name, module in model.named_modules():
        # 这里只捕获 freq_encoder 里的 StrongFreqBlock
        if isinstance(module, StrongFreqBlock):
            # hook 1: 整个 block，拿输入做 FFT，拿输出的 low/mid/high 做 map
            def make_freq_block_hook(layer_name):
                def hook(mod, inputs, outputs):
                    with torch.no_grad():
                        x_in = inputs[0].detach().float()  # [B,C,H,W]
                        B, C, H, W = x_in.shape
                        x_ch = x_in[0, 0]  # [H,W]
                        X = torch.fft.fftshift(torch.fft.fft2(x_ch, norm="ortho"))
                        mag = torch.log1p(torch.abs(X))      # [H,W]
                        radial = _radial_profile(mag.cpu())  # [R]

                        debug_state["freq_mag"].append(
                            (layer_name, mag.cpu(), radial)
                        )

                        # outputs: x_fused, x_low, x_mid, x_high
                        if isinstance(outputs, (tuple, list)) and len(outputs) == 4:
                            _, low, mid, high = outputs
                            low_map = low[0].detach().float().mean(dim=0).cpu()   # [H,W]
                            mid_map = mid[0].detach().float().mean(dim=0).cpu()
                            high_map = high[0].detach().float().mean(dim=0).cpu()
                            debug_state["freq_prior"].append(
                                (layer_name, "low", low_map)
                            )
                            debug_state["freq_prior"].append(
                                (layer_name, "mid", mid_map)
                            )
                            debug_state["freq_prior"].append(
                                (layer_name, "high", high_map)
                            )
                return hook

            handles.append(module.register_forward_hook(make_freq_block_hook(name)))

            # hook 2: 对应的 freq_gate，拿三频带的通道概率
            if hasattr(module, "freq_gate") and isinstance(module.freq_gate, nn.Module):
                def make_freq_gate_hook(layer_name):
                    def hook(mod, inputs, output):
                        with torch.no_grad():
                            out = output.detach()  # [B,3C,1,1]
                            B2, C3, _, _ = out.shape
                            num_bands = 3
                            if C3 % num_bands != 0:
                                return
                            C = C3 // num_bands
                            out_view = out.view(B2, num_bands, C)  # [B,3,C]
                            probs = torch.softmax(out_view, dim=1)  # [B,3,C]
                            probs_mean = probs.mean(dim=(0, 2))    # [3]
                            debug_state["freq_gate"].append(
                                (layer_name, probs_mean.cpu())
                            )
                    return hook

                handles.append(
                    module.freq_gate.register_forward_hook(make_freq_gate_hook(name))
                )

    # ------------------------------------------------------------------
    # 2) 前向一次，触发 hook，更新 MambaPriorUp 的 last_* 缓存
    # ------------------------------------------------------------------
    with torch.no_grad():
        _ = model(x)

    # hook 用完要移除
    for h in handles:
        h.remove()

    # ------------------------------------------------------------------
    # 3) 绘制 StrongFreqBlock 相关图像
    # ------------------------------------------------------------------
    # 3.1 频谱 + 径向曲线
    for idx, (layer_name, mag, radial) in enumerate(debug_state["freq_mag"]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax0, ax1 = axes

        im = ax0.imshow(mag.numpy(), cmap="viridis")
        ax0.set_title(f"Freq magnitude\n{layer_name}")
        ax0.set_xticks([])
        ax0.set_yticks([])
        fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

        r = np.linspace(0, 1, len(radial))
        ax1.plot(r, radial / (radial.max() + 1e-8))
        ax1.set_xlabel("Normalized radius (0→low, 1→high)")
        ax1.set_ylabel("Normalized energy")
        ax1.set_title("Radial frequency profile")

        fig.tight_layout()
        save_path = os.path.join(
            analysis_dir, f"freq_mag_radial_{idx}_{layer_name.replace('.', '_')}.png"
        )
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        made_any_figure = True

    # 3.2 三频带 gate 柱状图
    for idx, (layer_name, probs) in enumerate(debug_state["freq_gate"]):
        probs_np = probs.numpy()  # [3]
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        labels = ["low", "mid", "high"]
        ax.bar(labels, probs_np)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Average band probability")
        ax.set_title(f"Freq gates ({layer_name})")
        fig.tight_layout()
        save_path = os.path.join(
            analysis_dir, f"freq_gate_{idx}_{layer_name.replace('.', '_')}.png"
        )
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        made_any_figure = True

    # 3.3 三频带先验 map（低/中/高频）
    for idx, (layer_name, band_name, prior_map) in enumerate(debug_state["freq_prior"]):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        im = ax.imshow(prior_map.numpy(), cmap="magma")
        ax.set_title(f"{layer_name} - {band_name} prior (mean over C)")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        save_path = os.path.join(
            analysis_dir,
            f"freq_prior_{idx}_{layer_name.replace('.', '_')}_{band_name}.png",
        )
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        made_any_figure = True

    # ------------------------------------------------------------------
    # 4) 绘制解码块 MambaPriorUp 的路由 & 先验使用情况
    # ------------------------------------------------------------------
    for idx, up in enumerate(
        [getattr(model, "up1", None),
         getattr(model, "up2", None),
         getattr(model, "up3", None)],
        start=1
    ):
        if up is None:
            continue

        # 4.1 路由权重 alpha (low/mid/high)
        if getattr(up, "last_alpha", None) is not None:
            alpha = up.last_alpha[0].detach().cpu().numpy()  # [3]
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            labels = ["low", "mid", "high"]
            ax.bar(labels, alpha)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Routing weight α")
            ax.set_title(f"Decoder up{idx} routing α")
            fig.tight_layout()
            save_path = os.path.join(analysis_dir, f"up{idx}_routing_alpha.png")
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            made_any_figure = True

        # 4.2 空间 gate β(x)
        if getattr(up, "last_beta", None) is not None:
            beta = up.last_beta[0, 0].detach().cpu().numpy()  # [H,W]
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            im = ax.imshow(beta, cmap="inferno")
            ax.set_title(f"up{idx} spatial gate β(x)")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            save_path = os.path.join(analysis_dir, f"up{idx}_beta_gate.png")
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            made_any_figure = True

        # 4.3 prior_mix 热力图
        if getattr(up, "last_prior_mix", None) is not None:
            prior_mix = up.last_prior_mix[0].detach()  # [C,H,W]
            prior_mix_map = prior_mix.pow(2).mean(dim=0).cpu().numpy()  # [H,W]
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            im = ax.imshow(prior_mix_map, cmap="magma")
            ax.set_title(f"up{idx} prior_mix energy (mean over C)")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            save_path = os.path.join(analysis_dir, f"up{idx}_prior_mix.png")
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            made_any_figure = True

        if getattr(up, "last_out", None) is not None:
            out_feat = up.last_out[0].detach()  # [C,H,W]
            energy = out_feat.pow(2).mean(dim=0).cpu().numpy()
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            im = ax.imshow(energy, cmap="inferno")
            ax.set_title(f"up{idx} output energy (mean over C)")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            save_path = os.path.join(analysis_dir, f"up{idx}_out_energy.png")
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            made_any_figure = True

        # 4.4 分别画 prior_low / prior_mid / prior_high（可选）
        for band_name, tensor in [
            ("low", getattr(up, "last_prior_low", None)),
            ("mid", getattr(up, "last_prior_mid", None)),
            ("high", getattr(up, "last_prior_high", None)),
        ]:
            if tensor is None:
                continue
            pm = tensor[0].detach().float().mean(dim=0).cpu().numpy()  # [H,W]
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            im = ax.imshow(pm, cmap="viridis")
            ax.set_title(f"up{idx} {band_name} prior (mean over C)")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            save_path = os.path.join(
                analysis_dir, f"up{idx}_prior_{band_name}.png"
            )
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            made_any_figure = True

    if made_any_figure:
        print(f"📷 新版内部可视化已保存到: {analysis_dir}")
    else:
        print(f"⚠️ visualize_internal 没有生成任何图（检查 StrongFreqBlock hook 和 MambaPriorUp.last_* 是否生效）")



# ==================== 解析命令行参数 ====================
parser = argparse.ArgumentParser("SAM2-UNet Training")
parser.add_argument("--hiera_path", type=str, required=True, help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True,
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True, help="path to the mask file for training")
parser.add_argument("--val_ratio", type=float, default=0.1, help="validation set ratio")
parser.add_argument('--save_path', type=str, required=True, help="path to store the checkpoint and results")
parser.add_argument("--epoch", type=int, default=100, help ="training epochs")
parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int, help="batch size for training and validation")
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--patience", type=int, default=50, help="early stopping patience")
parser.add_argument("--min_delta", type=float, default=0.0005, help="minimum delta for early stopping")
parser.add_argument("--resume", type=str, default="", help="path to checkpoint for resuming training")
parser.add_argument("--save_val_interval", type=int, default=3, help="save validation results & analysis every N epochs")
parser.add_argument("--max_val_samples", type=int, default=20, help="max validation samples to save per epoch")
args = parser.parse_args()


def main(args):
    # 1. 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 2. 准备数据集
    full_dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')

    # 划分训练集和验证集
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=val_collate_fn
    )

    # 3. 初始化模型、优化器和学习率调度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAM2UNet(args.hiera_path).to(device)

    start_lr = args.lr
    optimizer = opt.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=start_lr,
        weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # 4. 初始化混合精度训练
    scaler = None
    if AMP_AVAILABLE and torch.cuda.is_available():
        scaler = GradScaler()
    else:
        print("⚠️ 使用标准精度训练")

    # 5. 训练准备
    os.makedirs(args.save_path, exist_ok=True)
    val_results_dir = os.path.join(args.save_path, "validation_results")
    os.makedirs(val_results_dir, exist_ok=True)

    best_dice = 0.0
    early_stop_counter = 0
    train_losses = []
    val_metrics = {
        'dice': [], 'iou': [], 'precision': [], 'recall': [],
        'f1': [], 'specificity': [], 'accuracy': []
    }

    # 断点重连机制
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> 从检查点 '{args.resume}' 恢复训练")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint.get('epoch', 0)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_dice = checkpoint.get('best_dice', 0.0)
            train_losses = checkpoint.get('train_losses', [])
            val_metrics = checkpoint.get('val_metrics', val_metrics)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"=> 成功加载检查点 (epoch {start_epoch})")
        else:
            print(f"=> 未找到检查点 '{args.resume}'")

    # 6. 训练循环
    print(f"\n🚀 开始训练，总 epochs: {args.epoch}, 初始学习率: {start_lr:.6f}, Batch Size: {args.batch_size}")
    for epoch in range(start_epoch, args.epoch):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epoch}')

        for batch in progress_bar:
            # 载入数据（先在 CPU 上做增强，再搬到 GPU）
            x = batch['image']
            target = batch['label']

            # 应用数据增强
            x, target = apply_augmentation(x, target)

            # 移到 GPU
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()

            def forward_and_loss(x, target):
                """前向 + 损失计算（同时用于 AMP 和非 AMP 分支）"""
                pred0, pred1, pred2 = model(x)

                if check_nan_inf(pred0, "pred0") or \
                   check_nan_inf(pred1, "pred1") or \
                   check_nan_inf(pred2, "pred2"):
                    return None, True  # loss=None, has_nan=True

                # 主输出 + 辅助输出（深监督）
                loss_main = brain_tumor_loss(pred0, target, freq_weight=0.02,boundary_weight=0.05)
                loss_aux2 = brain_tumor_loss(pred1, target, freq_weight=0.02,boundary_weight=0.05)
                loss_aux3 = brain_tumor_loss(pred2, target, freq_weight=0.02,boundary_weight=0.05)

                seg_loss = loss_main + 0.4 * (loss_aux2 + loss_aux3)

                # IFM 正则（如果你在 SAM2UNet 里实现了 get_ifm_reg_loss）
                ifm_reg = model.get_ifm_reg_loss()
                loss = seg_loss + ifm_reg

                if check_nan_inf(loss, "loss"):
                    return None, True

                return loss, False

            # ========= AMP 分支 =========
            if scaler is not None:
                with autocast():
                    loss, bad = forward_and_loss(x, target)

                if bad or loss is None:
                    print("跳过此批次更新因为检测到 NaN/Inf")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            # ========= 标准 FP32 =========
            else:
                loss, bad = forward_and_loss(x, target)

                if bad or loss is None:
                    print("跳过此批次更新因为检测到 NaN/Inf")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 统计与日志
            epoch_loss += loss.item()
            train_losses.append(loss.item())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # 是否保存当前 epoch 的验证可视化结果
        save_results_this_epoch = (epoch + 1) % args.save_val_interval == 0

        # 验证
        val_metrics_epoch = validate(
            model,
            val_loader,
            device,
            epoch + 1,  # 使用 1-based 索引
            val_results_dir,
            save_results=save_results_this_epoch,
            max_val_samples=args.max_val_samples
        )

        # 记录所有验证指标
        for key, value in val_metrics_epoch.items():
            if key not in val_metrics:
                val_metrics[key] = []
            val_metrics[key].append(value)

        # 使用 Dice 作为学习率调度器的指标
        scheduler.step(val_metrics_epoch['dice'])

        # 打印详细 epoch 结果
        print(f'\n📊 Epoch {epoch + 1}/{args.epoch} 详细指标 (LR: {optimizer.param_groups[0]["lr"]:.6f}):')
        print(f'   🎯 Train Loss: {epoch_loss / max(len(train_loader), 1):.4f}')
        print(f'   📈 Val Dice: {val_metrics_epoch["dice"]:.4f} (Best: {best_dice:.4f})')
        print(f'   🎯 Val IoU: {val_metrics_epoch["iou"]:.4f}')
        print(f'   🔍 Val Precision: {val_metrics_epoch["precision"]:.4f}')
        print(f'   📊 Val Recall: {val_metrics_epoch["recall"]:.4f}')
        print(f'   🎪 Val F1-Score: {val_metrics_epoch["f1"]:.4f}')
        print(f'   🛡️  Val Specificity: {val_metrics_epoch["specificity"]:.4f}')
        print(f'   ✅ Val Accuracy: {val_metrics_epoch["accuracy"]:.4f}')

        if save_results_this_epoch:
            print(f'   📸 验证结果已保存到: {os.path.join(val_results_dir, f"val_results_epoch_{epoch + 1}")}')
            # 新增：做一次内部频域可视化
            visualize_internal(
                model,
                val_loader,
                device,
                args.save_path,
                epoch + 1
            )

        # 保存最佳模型
        current_dice = val_metrics_epoch['dice']
        if current_dice > best_dice + args.min_delta:
            best_dice = current_dice
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
            print(f'💾 New best model saved with Dice: {best_dice:.4f}')
        else:
            early_stop_counter += 1

        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_path, f'epoch_{epoch + 1}.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'train_losses': train_losses,
                'val_metrics': val_metrics,
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"保存检查点到 {checkpoint_path}")

        # 早停检查
        if early_stop_counter >= args.patience:
            print(f'\nEarly stopping triggered at epoch {epoch + 1}')
            break

    # 7. 训练结束，保存最终结果
    plot_training_curve(train_losses, val_metrics, args.save_path, args)

    final_checkpoint = {
        'epoch': args.epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_dice': best_dice,
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None
    }
    torch.save(final_checkpoint, os.path.join(args.save_path, 'final_model.pth'))

    print(f'\nTraining completed. Best Val Dice: {best_dice:.4f}')


if __name__ == "__main__":
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f'Training started at {start_time}')

    main(args)

    end_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f'Training completed at {end_time}')
