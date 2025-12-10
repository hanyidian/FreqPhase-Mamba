# visualize_single_sample.py
import os
import sys
import argparse
import warnings
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 美观一点的全局画图风格
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_wavelets")

# ----------------- 路径 & 导入项目模块 -----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')
if os.path.exists(models_dir):
    sys.path.insert(0, models_dir)

try:
    # 按你现在的命名来，如果文件名不同这里改成 SAM2UNetXX
    from SAM2UNet17 import SAM2UNet, StrongFreqBlock, InformationFilter
    from dataset import FullDataset
    from utils import compute_metrics
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("请确认 SAM2UNet17.py / dataset.py / utils.py 在同一工程路径下。")
    sys.exit(1)


# ----------------- 一些小工具函数 -----------------
def ensure_rgb(img_tensor):
    """(C,H,W) -> (3,H,W)，方便画图"""
    if img_tensor.size(0) == 1:
        return img_tensor.repeat(3, 1, 1)
    elif img_tensor.size(0) == 3:
        return img_tensor
    else:
        # 如果是别的通道数，取前三个/或者重复第一通道
        if img_tensor.size(0) > 3:
            return img_tensor[:3]
        else:
            return img_tensor[0:1].repeat(3, 1, 1)


def save_segmentation_triplet(image, mask, pred, save_path):
    """
    原图 / GT / 预测 三联图，适合直接放论文。
    image, mask, pred: (C,H,W), 未必是0~1，这里会自己规范化。
    """
    img = image.detach().cpu()
    msk = mask.detach().cpu()
    prd = pred.detach().cpu()

    img = ensure_rgb(img)
    msk = msk.squeeze(0)
    prd = prd.squeeze(0)

    # 归一化到 0~1
    img_np = img.numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    msk_np = msk.numpy()
    prd_np = prd.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    ax1, ax2, ax3 = axes

    ax1.imshow(np.transpose(img_np, (1, 2, 0)))
    ax1.set_title("Input image")
    ax1.axis("off")

    ax2.imshow(msk_np, cmap="gray")
    ax2.set_title("Ground truth")
    ax2.axis("off")

    ax3.imshow(prd_np, cmap="gray")
    ax3.set_title("Prediction")
    ax3.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_freq_mag(mag, title, save_path):
    """频谱热力图（log幅度），去掉坐标轴，适合论文。"""
    mag_np = mag.cpu().numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mag_np, cmap="viridis")
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_bar(values, labels, title, save_path, ylim=(0, 1.0)):
    values = values.cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(labels, values)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_mask_heatmap(mask_2d, title, save_path):
    """InformationFilter 的空间 mask 热力图"""
    m = mask_2d.cpu().numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(m, cmap="magma")
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ----------------- 核心：单图可视化 -----------------
def visualize_single_sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 构建模型并加载 checkpoint
    print("🔧 Building model & loading checkpoint ...")
    model = SAM2UNet(args.hiera_path).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 2. 构建 test dataset，取一张图
    print("🖼  Loading single sample from dataset ...")
    dataset = FullDataset(args.image_path, args.mask_path, args.img_size, mode="test")
    if len(dataset) == 0:
        print("❌ Test dataset is empty.")
        return

    idx = max(0, min(args.index, len(dataset) - 1))
    sample = dataset[idx]
    img = sample["image"].unsqueeze(0).to(device)   # (1,C,H,W)
    msk = sample["label"].unsqueeze(0).to(device)   # (1,1,H,W)

    # 3. 注册 hook，收集频域 & gate & mask 信息
    debug = defaultdict(list)
    handles = []

    # ---- StrongFreqBlock: 输入频谱 + 3-band gate ----
    for name, module in model.named_modules():
        if isinstance(module, StrongFreqBlock):

            def make_freq_hook(layer_name):
                def hook(mod, inputs, output):
                    x_in = inputs[0].detach().float()    # [B,C,H,W]
                    # 只看第一个样本的第一个通道
                    x_ch = x_in[0, 0]
                    X = torch.fft.fftshift(torch.fft.fft2(x_ch, norm="ortho"))
                    mag = torch.log1p(torch.abs(X))
                    debug["freq_mag"].append((layer_name, mag.cpu()))
                return hook

            h1 = module.register_forward_hook(make_freq_hook(name))
            handles.append(h1)

            # 对 freq_gate 的输出做 3-band softmax 统计
            if hasattr(module, "freq_gate") and isinstance(module.freq_gate, torch.nn.Module):

                def make_gate_hook(layer_name):
                    def hook(mod, inputs, output):
                        # output: [B,3C,1,1]
                        out = output.detach()
                        B2, C3, _, _ = out.shape
                        num_bands = 3
                        assert C3 % num_bands == 0, "freq_gate 输出通道数应为 3*C"
                        C = C3 // num_bands
                        out = out.view(B2, num_bands, C)   # [B,3,C]
                        probs = torch.softmax(out, dim=1)  # 在 3 个 band 上做 softmax
                        probs_mean = probs.mean(dim=(0, 2))  # → [3]
                        debug["freq_gate"].append((layer_name, probs_mean.cpu()))
                    return hook

                h2 = module.freq_gate.register_forward_hook(make_gate_hook(name))
                handles.append(h2)

    # ---- InformationFilter: 分支 gate + 空间 mask ----
    for name, module in model.named_modules():
        if isinstance(module, InformationFilter):

            # 分支 gate (SAM / Mamba / Freq)
            if hasattr(module, "branch_gate") and isinstance(module.branch_gate, torch.nn.Module):

                def make_branch_hook(layer_name):
                    def hook(mod, inputs, output):
                        # output: [B,3,1,1]
                        out = output.detach()
                        out = out.view(out.size(0), 3)     # [B,3]
                        probs = torch.softmax(out, dim=1)  # [B,3]
                        probs_mean = probs.mean(dim=0)     # [3]
                        debug["branch_gate"].append((layer_name, probs_mean.cpu()))
                    return hook

                h3 = module.branch_gate.register_forward_hook(make_branch_hook(name))
                handles.append(h3)

            # 空间 mask（local_fuse 输出）
            if hasattr(module, "local_fuse") and isinstance(module.local_fuse, torch.nn.Module):

                def make_mask_hook(layer_name):
                    def hook(mod, inputs, output):
                        # output: [B,C,H,W]
                        mask = output.detach()
                        mask_mean = mask.mean(dim=1)[0]    # 通道均值后取第一个样本 -> [H,W]
                        debug["if_mask"].append((layer_name, mask_mean.cpu()))
                    return hook

                h4 = module.local_fuse.register_forward_hook(make_mask_hook(name))
                handles.append(h4)

    # 4. 前向传播（触发 hook）并得到预测
    print("🚀 Running forward pass with hooks ...")
    with torch.no_grad():
        pred, _, _ = model(img)

    # 预测概率 -> 0/1
    prob = torch.sigmoid(pred)
    pred_bin = (prob > 0.5).float()

    # 5. 保存 segmentation 三联图
    save_segmentation_triplet(
        img[0].cpu(), msk[0].cpu(), pred_bin[0].cpu(),
        os.path.join(args.save_dir, "seg_triplet.png")
    )
    print(f"✅ Saved segmentation triplet to {os.path.join(args.save_dir, 'seg_triplet.png')}")

    # 6. 开始画各种内部可视化
    # 6.1 频谱图
    for idx, (layer_name, mag) in enumerate(debug["freq_mag"]):
        fname = f"freq_mag_{idx}_{layer_name.replace('.', '_')}.png"
        title = f"Freq magnitude - {layer_name}"
        save_freq_mag(mag, title, os.path.join(args.save_dir, fname))

    # 6.2 3-band 频带 gate
    for idx, (layer_name, probs) in enumerate(debug["freq_gate"]):
        fname = f"freq_gate_{idx}_{layer_name.replace('.', '_')}.png"
        title = f"Freq gates (low/mid/high) - {layer_name}"
        labels = ["low", "mid", "high"]
        save_bar(probs, labels, title, os.path.join(args.save_dir, fname))

    # 6.3 信息过滤模块：分支 gate
    for idx, (layer_name, probs) in enumerate(debug["branch_gate"]):
        fname = f"if_branch_gate_{idx}_{layer_name.replace('.', '_')}.png"
        title = f"Branch gates - {layer_name}"
        labels = ["sem(SAM)", "Mamba", "Freq"]
        save_bar(probs, labels, title, os.path.join(args.save_dir, fname))

    # 6.4 信息过滤模块：空间 mask
    for idx, (layer_name, mask_2d) in enumerate(debug["if_mask"]):
        fname = f"if_mask_{idx}_{layer_name.replace('.', '_')}.png"
        title = f"Information mask - {layer_name}"
        save_mask_heatmap(mask_2d, title, os.path.join(args.save_dir, fname))

    # 7. 简单打印一下预测指标（只是 sanity check）
    metrics = compute_metrics(pred, msk)
    print("\n📊 Metrics on this single sample:")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}")

    # 8. 清理 hooks
    for h in handles:
        h.remove()

    print(f"\n✨ All visualizations saved under: {args.save_dir}")


# ----------------- main & argparse -----------------
def parse_args():
    parser = argparse.ArgumentParser("Single-sample visualization for SAM2UNet17")
    parser.add_argument("--hiera_path", type=str, required=True, help="Path to sam2 pretrained hiera .pt")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--image_path", type=str, required=True, help="Test images folder")
    parser.add_argument("--mask_path", type=str, required=True, help="Test masks folder")
    parser.add_argument("--save_dir", type=str, required=True, help="Folder to save visualizations")
    parser.add_argument("--img_size", type=int, default=352, help="Resize size used in your dataset")
    parser.add_argument("--index", type=int, default=0, help="Which sample index in test set to visualize")
    return parser.parse_args()
# --hiera_path "/root/autodl-tmp/checkpoints/sam2.1_hiera_large.pt" --train_image_path "/root/autodl-tmp/datasets/CVC-ClinicDB/PNG/data/train/images" --train_mask_path "/root/autodl-tmp/datasets/CVC-ClinicDB/PNG/data/train/masks" --save_path "/root/autodl-tmp/checkpoints/1/base"

if __name__ == "__main__":
    args = parse_args()
    visualize_single_sample(args)
