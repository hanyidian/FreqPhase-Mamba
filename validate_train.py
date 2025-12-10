import os
import sys
import glob
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import warnings
from collections import namedtuple

# 忽略特定警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_wavelets")

# ==============================================================================
# ## CONFIGURATION ##
# 你需要修改的所有参数都在这里
# ==============================================================================

# 📊 最佳模型的详细性能指标:
# ==================================================
#    Dice        : 0.9226
#    Iou         : 0.8581
#    Precision   : 0.9276
#    Recall      : 0.9213
#    F1          : 0.9226
#    Specificity : 0.9812
#    Accuracy    : 0.9746
# ==================================================
# 1. 包含所有.pth检查点文件的文件夹路径
CHECKPOINTS_DIR = "/root/autodl-tmp/checkpoints/2"

# 2. Hiera模型文件路径
HIERA_PATH = "/root/autodl-tmp/checkpoints/sam2.1_hiera_large.pt"

# 3. 测试数据集路径
TEST_IMAGE_PATH = "/root/autodl-tmp/datasets/ISIC/test/images"
TEST_MASK_PATH = "/root/autodl-tmp/datasets/ISIC/test/labels"
# TEST_IMAGE_PATH = "/root/autodl-tmp/datasets/Kvasir-SEG/Kvasir-SEG/data/test/images"
# TEST_MASK_PATH = "/root/autodl-tmp/datasets/Kvasir-SEG/Kvasir-SEG/data/test/masks"
# TEST_IMAGE_PATH = "/root/autodl-tmp/datasets/total/ETIS-LaribPolypDB/data/test/im ages"
# TEST_IMAGE_PATH = "/root/autodl-tmp/datasets/CVC-ClinicDB/PNG/data/test/images"
# TEST_MASK_PATH = "/root/autodl-tmp/datasets/CVC-ClinicDB/PNG/data/test/masks"
# TEST_IMAGE_PATH = "/root/autodl-tmp/datasets/total/CVC-300/data/test/images"
# TEST_MASK_PATH = "/root/autodl-tmp/datasets/total/CVC-300/data/test/labels"
# TEST_IMAGE_PATH = "/root/autodl-tmp/datasets/total/CVC-ColonDB/data/test/images"
# TEST_MASK_PATH = "/root/autodl-tmp/datasets/total/CVC-ColonDB/data/test/labels"
# TEST_IMAGE_PATH = "/root/autodl-tmp/datasets/total/ETIS-LaribPolypDB/data/test/images"
# TEST_MASK_PATH = "/root/autodl-tmp/datasets/total/ETIS-LaribPolypDB/data/test/labels"
# 4. 测试参数
BATCH_SIZE = 8
IMG_SIZE = 352

# 5. (可选) 保存可视化结果的路径
#    设置为一个文件夹路径 (例如 "/root/autodl-tmp/test_results") 来保存图片。
#    设置为 None 则不保存。
OUTPUT_RESULTS_DIR = None
MAX_TEST_SAMPLES_TO_SAVE = 10  # 如果保存，每个模型最多保存多少个样本

# ==============================================================================
# ## SCRIPT CODE ##
# 从这里开始，你通常不需要修改代码
# ==============================================================================

# --- 动态添加项目路径 (如果需要) ---
# 确保可以导入项目中的其他.py文件 (如模型定义、数据集等)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设 models, dataset.py, utils.py 与此脚本在同一层级或可访问
models_dir = os.path.join(current_dir, 'models')
if os.path.exists(models_dir):
    sys.path.insert(0, models_dir)

try:
    from model9 import SAM2UNet
    from dataset import FullDataset
    from utils import compute_metrics, val_collate_fn
    from torchvision.utils import save_image
except ImportError as e:
    print(f"错误: 无法导入必要的模块: {e}")
    print("请确保 SAM2UNet5.py, dataset.py, utils.py 文件在Python路径中，或与此脚本在同一目录下。")
    sys.exit(1)


def save_test_visual_results(images, masks, preds, batch_indices, output_dir):
    """保存测试结果图像。"""
    os.makedirs(output_dir, exist_ok=True)
    preds_bin = (torch.sigmoid(preds) > 0.5).float()

    for i in range(images.size(0)):
        img, mask, pred = images[i].cpu(), masks[i].cpu(), preds_bin[i].cpu()
        if img.size(0) == 1: img = img.repeat(3, 1, 1)
        if mask.size(0) == 1: mask = mask.repeat(3, 1, 1)
        if pred.size(0) == 1: pred = pred.repeat(3, 1, 1)

        combined = torch.cat([img, mask, pred], dim=2)
        sample_idx = batch_indices[i] if batch_indices is not None and len(batch_indices) > i else i
        save_image(combined, os.path.join(output_dir, f"test_sample_{sample_idx}_combined.png"))


def run_test_for_checkpoint(args):
    """
    为单个检查点执行测试的核心函数。
    接收一个包含所有参数的命名空间对象 (args)。
    返回一个包含所有评估指标的字典。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    model = SAM2UNet(args.hiera_path).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)

    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()

    # 2. 准备测试数据集
    test_dataset = FullDataset(args.test_image_path, args.test_mask_path, args.img_size, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                             pin_memory=True, collate_fn=val_collate_fn)

    if not test_dataset:
        print("错误: 测试数据集为空。")
        return None

    # 3. 进行评估
    metrics_sum = {k: 0 for k in ['dice', 'iou', 'precision', 'recall', 'f1', 'specificity', 'accuracy']}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {os.path.basename(args.model_path)}", leave=False):
            x, target = batch['image'].to(device), batch['label'].to(device)
            # pred, _, _, = model(x)
            pred, _, _ = model(x)

            batch_metrics = compute_metrics(pred, target)
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]

    if not test_loader:
        print("没有样本被评估。")
        return None

    # 4. 计算平均指标
    num_batches = len(test_loader)
    avg_metrics = {key: value / num_batches for key, value in metrics_sum.items()}

    return avg_metrics


def main():
    """
    主执行函数：查找并评估所有检查点，找出最佳模型。
    """
    # 1. 查找所有.pth文件
    checkpoint_paths = glob.glob(os.path.join(CHECKPOINTS_DIR, "*.pth")) + glob.glob(
        os.path.join(CHECKPOINTS_DIR, "*.pt"))

    if not checkpoint_paths:
        print(f"错误：在目录 '{CHECKPOINTS_DIR}' 中没有找到任何 .pth 或 .pt 文件。")
        return

    print(f"在 '{CHECKPOINTS_DIR}' 中找到 {len(checkpoint_paths)} 个检查点文件，开始逐一评估...")
    print("-" * 70)

    best_dice = -1.0
    best_checkpoint_path = None
    best_metrics = None

    # 使用命名元组创建一个类似argparse的参数对象
    TestArgs = namedtuple('TestArgs', [
        'hiera_path', 'test_image_path', 'test_mask_path', 'batch_size',
        'img_size', 'output_results_dir', 'max_test_samples_to_save', 'model_path'
    ])

    # 2. 遍历并测试每个检查点
    for i, ckpt_path in enumerate(sorted(checkpoint_paths)):  # sorted()保证测试顺序
        print(f"[{i + 1}/{len(checkpoint_paths)}] 正在准备测试: {os.path.basename(ckpt_path)}")

        current_args = TestArgs(
            hiera_path=HIERA_PATH,
            test_image_path=TEST_IMAGE_PATH,
            test_mask_path=TEST_MASK_PATH,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            output_results_dir=OUTPUT_RESULTS_DIR,
            max_test_samples_to_save=MAX_TEST_SAMPLES_TO_SAVE,
            model_path=ckpt_path
        )

        metrics = run_test_for_checkpoint(current_args)

        if metrics is None:
            print(f"测试失败: {os.path.basename(ckpt_path)}. 跳过。")
            print("-" * 70)
            continue

        current_dice = metrics.get('dice', -1.0)
        print(f"✅ 完成测试: {os.path.basename(ckpt_path)} -> Dice Score: {current_dice:.4f}")

        # 3. 比较并记录最佳结果
        if current_dice > best_dice:
            print(f"🏆 新的最佳模型！ Dice: {current_dice:.4f} (之前最佳: {best_dice:.4f})")
            best_dice = current_dice
            best_checkpoint_path = ckpt_path
            best_metrics = metrics

        print("-" * 70)

    # 4. 最终报告
    if best_checkpoint_path:
        print("\n======================= 最终评估完成 =======================")
        print(f"🏆 最佳模型文件是: {os.path.basename(best_checkpoint_path)}")
        print(f"   位于: {best_checkpoint_path}")
        print("\n📊 最佳模型的详细性能指标:")
        print('=' * 50)
        for key, value in best_metrics.items():
            print(f'   {key.replace("_", " ").capitalize():<12}: {value:.4f}')
        print('=' * 50)
    else:
        print("\n所有检查点的测试都失败了或没有找到检查点，无法确定最佳模型。")


if __name__ == "__main__":
    main()
