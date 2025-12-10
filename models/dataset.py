import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ToTensor(object):
    """将PIL图像转换为PyTorch张量"""

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):
    """调整图像和标签的大小"""

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        # 注意: 对于标签，通常使用最近邻插值 (NEAREST) 来避免引入新的像素值。
        # BICUBIC可能会产生0-1之外的值，导致二值标签不再是纯0/1。
        # 如果你的标签是连续的（如深度图），BICUBIC没问题。
        # 但息肉分割是二值标签，建议改为NEAREST。
        return {'image': F.resize(image, self.size, interpolation=InterpolationMode.BILINEAR),
                'label': F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)}


class RandomHorizontalFlip(object):
    """随机水平翻转"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    """随机垂直翻转"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    """标准化图像"""

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


class FullDataset(Dataset):
    """
    用于训练和验证的数据集类

    功能：
    1. 自动扫描图像和标签目录
    2. 配对图像和标签文件
    3. 根据训练/验证模式应用不同的数据增强
    4. 返回PyTorch张量格式的数据
    """

    def __init__(self, image_root, gt_root, size, mode):
        """
        Args:
            image_root: 图像文件夹路径
            gt_root: 标签(ground truth)文件夹路径
            size: 图像目标尺寸
            mode: 'train' 或 'val'
        """
        # --- 修改开始 ---
        # 使用 os.path.join 替换字符串拼接和硬编码的反斜杠
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root)
                      if f.endswith(('.jpg', '.png', '.tif', '.tiff', '.bmp', '.jpeg'))]

        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root)
                   if f.endswith(('.jpg', '.png', '.tif', '.tiff', '.bmp', '.jpeg'))]
        # --- 修改结束 ---

        # 排序确保配对正确
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # 检查文件数量是否匹配
        if len(self.images) != len(self.gts):
            raise ValueError(f"图像数量({len(self.images)})与标签数量({len(self.gts)})不匹配")

        print(f"找到 {len(self.images)} 对图像-标签文件")

        # 根据模式设置数据增强流水线
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),           # 调整大小
                RandomHorizontalFlip(p=0.5),    # 随机水平翻转
                RandomVerticalFlip(p=0.5),      # 随机垂直翻转
                ToTensor(),                     # 转换为张量
                Normalize()                     # 标准化
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),           # 调整大小
                ToTensor(),                     # 转换为张量
                Normalize()                     # 标准化
            ])

    def __getitem__(self, idx):
        """获取单个样本"""
        # 加载图像和标签
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])

        # 创建数据字典
        data = {'image': image, 'label': label}

        # 应用变换
        data = self.transform(data)

        return data

    def __len__(self):
        """返回数据集大小"""
        return len(self.images)

    def rgb_loader(self, path):
        """加载RGB图像"""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        """加载二值标签图像"""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class TestDataset:
    """
    用于测试的数据集类
    按顺序加载数据，不使用随机增强
    """

    def __init__(self, image_root, gt_root, size):
        # --- 修改开始 ---
        # 使用 os.path.join 替换字符串拼接和硬编码的反斜杠
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root)
                      if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root)
                   if f.endswith('.png') or f.endswith('.jpg')]
        # --- 修改结束 ---

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # 测试用变换(无数据增强)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
        # 注意: 对于标签，通常使用最近邻插值 (NEAREST) 来避免引入新的像素值。
        self.gt_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST), # 添加 Resize
            transforms.ToTensor()
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        """加载下一组测试数据"""
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0) # 添加 unsqueeze(0) 以创建 batch 维度

        gt = self.binary_loader(self.gts[self.index])
        gt_tensor = self.gt_transform(gt) # 应用 gt_transform
        gt = np.array(gt.resize((self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]))) # 获取 numpy 格式的 resize 后的 gt

        name = self.images[self.index].split(os.path.sep)[-1] # 使用 os.path.sep 获取正确的分隔符

        self.index += 1
        return image, gt, name, gt_tensor # 返回 gt_tensor 方便后续计算指标

    def rgb_loader(self, path):
        """加载RGB图像"""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        """加载二值标签图像"""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

