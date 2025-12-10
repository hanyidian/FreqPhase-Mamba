import torch
import torch.nn.functional as F
import os
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import numpy as np
import warnings

# 忽略pytorch_wavelets库可能产生的用户警告
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_wavelets")


def save_validation_results(images, masks, preds, epoch, save_path, sample_indices, max_val_samples):
    """
    保存验证结果图像（原始图像、真实掩码、预测掩码和组合图）。
    Args:
        images (Tensor): 验证图像批次。
        masks (Tensor): 真实掩码批次。
        preds (Tensor): 模型预测结果批次 (logits)。
        epoch (int): 当前训练epoch。
        save_path (str): 保存结果的根目录。
        sample_indices (list or None): 批次中每个样本在原始数据集中的索引，用于命名文件。
        max_val_samples (int): 每个epoch最多保存的验证样本数量。
    """
    epoch_dir = os.path.join(save_path, f"val_results_epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    preds_bin = (torch.sigmoid(preds) > 0.5).float()  # 将logits转换为二值预测

    for i in range(min(images.size(0), max_val_samples)):
        img = images[i].cpu()
        mask = masks[i].cpu()
        pred = preds_bin[i].cpu()

        # 确保图像是3通道以便保存
        if img.size(0) == 1:  # 如果是灰度图，复制到3通道
            img = img.repeat(3, 1, 1)
        if mask.size(0) == 1:
            mask = mask.repeat(3, 1, 1)
        if pred.size(0) == 1:
            pred = pred.repeat(3, 1, 1)

        # 组合图像：原始图 | 真实掩码 | 预测掩码
        combined = torch.cat([img, mask, pred], dim=2)
        sample_idx = sample_indices[i] if sample_indices is not None and len(sample_indices) > i else i

        save_image(combined, os.path.join(epoch_dir, f"sample_{sample_idx}_combined.png"))
        save_image(img, os.path.join(epoch_dir, f"sample_{sample_idx}_image.png"))
        save_image(mask, os.path.join(epoch_dir, f"sample_{sample_idx}_mask.png"))
        save_image(pred, os.path.join(epoch_dir, f"sample_{sample_idx}_pred.png"))

def _sobel_edge_map(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,1,H,W] 概率图或 0/1 mask
    返回归一化到 [0,1] 的边缘强度图
    """
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    kx = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + 1e-6)

    # 每张图单独归一化，避免 scale 过大
    g = g / (g.amax(dim=(-1, -2), keepdim=True) + 1e-6)
    return g

def brain_tumor_loss(
    logits,
    targets,
    bce_weight=0.5,
    tversky_weight=0.5,
    pos_weight=2.0,
    alpha=0.3,
    beta=0.7,
    freq_weight=0.02,       # ⭐ 建议：0.01~0.03
    boundary_weight=0.05,   # ⭐ 新增：边界一致性权重，先 0.03~0.06 试试
    ifm_reg=None,           # 可以传 model.get_ifm_reg_loss()
    eps=1e-6,
):
    """
    总损失 = Dice
          + bce_weight * BCE
          + tversky_weight * Tversky
          + boundary_weight * BoundaryAlign（Sobel 边缘一致性）
          + freq_weight * FreqAlign（频域 log-幅度对齐，偏重中高频）
          + ifm_reg（可选）

    说明：
      - 边界项主要帮你补高频/轮廓，对 decoder 里 high-band 更友好；
      - 频域项保持，但默认权重略调小，防止和结构上的 high-bias 冲突。
    """

    # 为了在 AMP 下也稳定，loss 统一用 float32 计算
    logits = logits.float()
    targets = targets.float()

    # -------- 1. BCE（带前景加权） --------
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets)

    # -------- 2. Soft Dice Loss --------
    probs = torch.sigmoid(logits)               # (B,1,H,W)
    probs_flat   = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    dice_loss = 1.0 - dice.mean()

    # -------- 3. Tversky Loss（偏惩 FN → 提升小目标召回） --------
    TP = (probs_flat * targets_flat).sum(dim=1)
    FP = (probs_flat * (1 - targets_flat)).sum(dim=1)
    FN = ((1 - probs_flat) * targets_flat).sum(dim=1)

    tversky_index = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    tversky_loss = 1.0 - tversky_index.mean()

    seg_loss = dice_loss + bce_weight * bce + tversky_weight * tversky_loss

    # =====================================
    # 4. 边界一致性正则（Sobel 边缘）BoundaryAlign
    # =====================================
    if boundary_weight > 0.0:
        pred_edge = _sobel_edge_map(probs)
        tgt_edge  = _sobel_edge_map(targets)

        boundary_reg = F.l1_loss(pred_edge, tgt_edge)
    else:
        boundary_reg = logits.new_tensor(0.0)

    # =====================================
    # 5. 频域对齐正则 FreqAlign（强调中高频）
    # =====================================
    if freq_weight > 0.0:
        # 为了避免 AMP 干扰 FFT，用 float32 显式计算
        probs_32 = probs.float()
        targets_32 = targets.float()

        # FFT 并做中心平移，便于构造径向权重
        probs_f = torch.fft.fftshift(torch.fft.fft2(probs_32, norm='ortho'))
        target_f = torch.fft.fftshift(torch.fft.fft2(targets_32, norm='ortho'))

        mag_pred = torch.log1p(torch.abs(probs_f))   # [B,1,H,W]
        mag_gt   = torch.log1p(torch.abs(target_f))

        B, C, H, W = mag_pred.shape
        yy, xx = torch.meshgrid(
            torch.arange(H, device=logits.device),
            torch.arange(W, device=logits.device),
            indexing='ij'
        )
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)    # [H,W]
        rr = rr / (rr.max() + 1e-6)                         # 归一化到 [0,1]

        # gamma 越大，高频越重；1.0 ~ 2.0 比较稳
        gamma = 1.5
        weight = rr ** gamma                                # [H,W]
        weight = weight[None, None, ...]                    # [1,1,H,W]

        # L1 距离 + 频率权重
        freq_reg = F.l1_loss(mag_pred * weight, mag_gt * weight)
    else:
        freq_reg = logits.new_tensor(0.0)

    # -------- 6. 汇总 --------
    loss = seg_loss \
           + boundary_weight * boundary_reg \
           + freq_weight * freq_reg

    # -------- 7. 可选：IFM 正则（来自 model.get_ifm_reg_loss()） --------
    if ifm_reg is not None:
        # ifm_reg 自己里面已经乘过 lambda_orth / lambda_entropy
        loss = loss + ifm_reg

    return loss

def compute_metrics(pred, gt):
    """
    计算医学图像分割的完整评估指标。
    Args:
        pred (Tensor): 模型预测的logits。
        gt (Tensor): 真实掩码 (0或1)。
    Returns:
        dict: 包含Dice, IoU, Precision, Recall, F1, Specificity, Accuracy的字典。
    """
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    gt = gt.float()

    intersection = (pred_bin * gt).sum()
    union = (pred_bin + gt).sum() - intersection
    pred_sum = pred_bin.sum()
    gt_sum = gt.sum()

    eps = 1e-7  # 避免除零

    dice = (2 * intersection) / (pred_sum + gt_sum + eps)
    iou = intersection / (union + eps)

    precision = intersection / (pred_sum + eps)
    recall = intersection / (gt_sum + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    total_pixels = pred_bin.numel()
    tn = ((1 - pred_bin) * (1 - gt)).sum()  # True Negatives
    fp = pred_sum - intersection  # False Positives
    fn = gt_sum - intersection  # False Negatives

    specificity = tn / (tn + fp + eps)
    accuracy = (intersection + tn) / (total_pixels + eps)

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'specificity': specificity.item(),
        'accuracy': accuracy.item()
    }


def validate(model, val_loader, device, epoch, save_path, save_results=True, max_val_samples=20):
    """
    验证模型在验证集上的性能，并可选择保存验证结果。
    Args:
        model (nn.Module): 待验证的模型。
        val_loader (DataLoader): 验证集数据加载器。
        device (torch.device): 设备 ('cuda' 或 'cpu')。
        epoch (int): 当前训练epoch (用于文件命名)。
        save_path (str): 保存验证结果图像的目录。
        save_results (bool): 是否保存验证结果图像。
        max_val_samples (int): 每个epoch最多保存的验证样本数量。
    Returns:
        dict: 包含平均Dice, IoU, Precision, Recall, F1, Specificity, Accuracy的字典。
    """
    model.eval()
    metrics_sum = {
        'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0,
        'f1': 0, 'specificity': 0, 'accuracy': 0
    }
    saved_samples_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            indices = batch.get('index', None)  # 获取样本索引
            pred, _, _ = model(x)  # 只使用主输出
            batch_metrics = compute_metrics(pred, target)
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]

            # 如果需要保存结果且未达到最大保存数量
            if save_results and saved_samples_count < max_val_samples:
                remaining_slots = max_val_samples - saved_samples_count
                samples_to_save_in_batch = min(x.size(0), remaining_slots)

                if samples_to_save_in_batch > 0:
                    save_validation_results(
                        x[:samples_to_save_in_batch],
                        target[:samples_to_save_in_batch],
                        pred[:samples_to_save_in_batch],
                        epoch,
                        save_path,
                        indices[:samples_to_save_in_batch] if indices is not None else None,
                        samples_to_save_in_batch  # 传入本次需要保存的数量
                    )
                    saved_samples_count += samples_to_save_in_batch

                    # 确保只保存一次，如果已达到最大数量，则停止进一步保存
                    if saved_samples_count >= max_val_samples:
                        save_results = False  # 将标志设为False，不再保存后续批次

    num_batches = len(val_loader)
    avg_metrics = {key: value / num_batches for key, value in metrics_sum.items()}
    return avg_metrics


def plot_training_curve(train_losses, val_metrics, save_path, args):
    fig = plt.figure(figsize=(20, 12))

    # 1. 训练损失
    plt.subplot(2, 4, 1)
    plt.plot(train_losses, label='Training Loss', color='red', linewidth=2)
    plt.title('Training Loss Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2. 核心分割指标
    plt.subplot(2, 4, 2)
    metrics_core = ['dice', 'iou', 'f1']
    colors_core = ['blue', 'green', 'purple']
    for i, metric in enumerate(metrics_core):
        if metric in val_metrics and val_metrics[metric]:
            plt.plot(val_metrics[metric], label=metric.capitalize(),
                     color=colors_core[i], linewidth=2, marker='o', markersize=4)
    plt.title('Core Segmentation Metrics', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # 3. 精确率和召回率
    plt.subplot(2, 4, 3)
    metrics_pr = ['precision', 'recall']
    colors_pr = ['orange', 'red']
    for i, metric in enumerate(metrics_pr):
        if metric in val_metrics and val_metrics[metric]:
            plt.plot(val_metrics[metric], label=metric.capitalize(),
                     color=colors_pr[i], linewidth=2, marker='s', markersize=4)
    plt.title('Precision & Recall', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # 4. 特异度和准确率
    plt.subplot(2, 4, 4)
    metrics_sa = ['specificity', 'accuracy']
    colors_sa = ['brown', 'gray']
    for i, metric in enumerate(metrics_sa):
        if metric in val_metrics and val_metrics[metric]:
            plt.plot(val_metrics[metric], label=metric.capitalize(),
                     color=colors_sa[i], linewidth=2, marker='^', markersize=4)
    plt.title('Specificity & Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # 5. Dice系数单独图表
    plt.subplot(2, 4, 5)
    if 'dice' in val_metrics and val_metrics['dice']:
        plt.plot(val_metrics['dice'], color='blue', linewidth=3, marker='o', markersize=6)
        plt.axhline(y=max(val_metrics['dice']), color='red', linestyle='--', alpha=0.7,
                    label=f'Best: {max(val_metrics["dice"]):.4f}')
    plt.title('Dice Coefficient', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # 6. IoU单独图表
    plt.subplot(2, 4, 6)
    if 'iou' in val_metrics and val_metrics['iou']:
        plt.plot(val_metrics['iou'], color='green', linewidth=3, marker='s', markersize=6)
        plt.axhline(y=max(val_metrics['iou']), color='red', linestyle='--', alpha=0.7,
                    label=f'Best: {max(val_metrics["iou"]):.4f}')
    plt.title('Intersection over Union', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # 7. 所有指标对比
    plt.subplot(2, 4, 7)
    metrics_all = ['dice', 'iou', 'precision', 'recall', 'f1']
    colors_all = ['blue', 'green', 'orange', 'red', 'purple']
    for i, metric in enumerate(metrics_all):
        if metric in val_metrics and val_metrics[metric]:
            plt.plot(val_metrics[metric], label=metric.capitalize(),
                     color=colors_all[i], linewidth=2, alpha=0.8)
    plt.title('All Metrics Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # 8. 训练概览
    plt.subplot(2, 4, 8)
    plt.axis('off')
    overview_text = f"""
🏥 医学影像分割训练概览

🎯 模型: SAM2-UNet (参数冻结优化)
⚡ 优化策略: 混合精度 + 参数冻结

📈 最佳性能指标:
• Dice: {max(val_metrics.get('dice', [0])):.4f}
• IoU: {max(val_metrics.get('iou', [0])):.4f}
• Precision: {max(val_metrics.get('precision', [0])):.4f}
• Recall: {max(val_metrics.get('recall', [0])):.4f}
• F1: {max(val_metrics.get('f1', [0])):.4f}
• Specificity: {max(val_metrics.get('specificity', [0])):.4f}
• Accuracy: {max(val_metrics.get('accuracy', [0])):.4f}

🔬 训练配置:
• Batch Size: {args.batch_size}
• Learning Rate: {args.lr}
• Epochs: {args.epoch}
• Device: {'GPU' if torch.cuda.is_available() else 'CPU'}
"""

    plt.text(0.1, 0.95, overview_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    plt.suptitle('🏥 SAM2-UNet 眼底血管分割训练监控面板', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 详细训练曲线已保存到: {os.path.join(save_path, 'training_curve.png')}")


def apply_augmentation(images, masks):
    """
    应用数据增强到图像和掩码批次 (使用Albumentations增强策略)。
    Args:
        images (Tensor): 原始图像批次 [B, C, H, W]。
        masks (Tensor): 原始掩码批次 [B, 1, H, W]。
    Returns:
        tuple: 增强后的图像批次和掩码批次。
    """
    try:
        import albumentations as A
    except ImportError:
        print("Albumentations not installed. Please install with: pip install albumentations")
        return images, masks

    # 定义增强流水线 (增强策略，保持稳定)
    transform = A.Compose([
    A.HorizontalFlip(p=0.6),
    A.VerticalFlip(p=0.6),
    A.Rotate(limit=30, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    # 新增：随机弹性变形 (低强度，低概率)
    A.ElasticTransform(alpha=50, sigma=5, p=0.3),  # 温和变形，模拟组织变化
    # 新增：随机Gamma调整 (低概率)
    # A.RandomGamma(gamma_limit=(90, 110), p=0.2),  # 模拟照明变化
])

    batch_size = images.size(0)
    augmented_images = []
    augmented_masks = []

    for i in range(batch_size):
        # 转换为Albumentations格式: [H, W, C] (先移到CPU)
        img_np = images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, C]
        mask_np = masks[i].cpu().squeeze().numpy()  # [H, W]

        # 应用增强
        augmented = transform(image=img_np, mask=mask_np)
        aug_img_np = augmented['image']
        aug_mask_np = augmented['mask']

        # 转回Tensor格式: [C, H, W]
        aug_img = torch.from_numpy(aug_img_np).permute(2, 0, 1).float()
        aug_mask = torch.from_numpy(aug_mask_np).unsqueeze(0).float()

        augmented_images.append(aug_img)
        augmented_masks.append(aug_mask)

    return torch.stack(augmented_images), torch.stack(augmented_masks)


def check_nan_inf(tensor, name):
    """
    检查张量中是否存在NaN或Inf值。
    Args:
        tensor (Tensor): 要检查的张量。
        name (str): 张量的名称，用于打印警告信息。
    Returns:
        bool: 如果存在NaN或Inf则返回True，否则返回False。
    """
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"WARNING: NaN or Inf detected in {name}")
        return True
    return False


def val_collate_fn(batch):
    """
    验证集数据整理函数，添加样本索引。
    Args:
        batch (list): 包含字典的列表，每个字典包含'image'和'label'。
    Returns:
        dict: 包含'image', 'label', 'index'的字典。
    """
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'index': [item.get('index', i) for i, item in enumerate(batch)]
    }

