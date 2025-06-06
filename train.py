# train.py
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models import EnhancedCNN
import time
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch.nn.functional as F
import cv2

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor
        
    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

# 数据增强
class RandomStretchRotation:
    """随机拉伸和旋转变换"""
    def __init__(self, max_rotation=45, max_stretch=0.3):
        self.max_rotation = max_rotation
        self.max_stretch = max_stretch
        
    def __call__(self, img):
        # 随机选择拉伸类型：水平、垂直或双向
        stretch_type = random.choice(['h', 'v', 'both'])
        
        # 随机拉伸因子
        h_stretch = 1.0
        v_stretch = 1.0
        
        if stretch_type == 'h' or stretch_type == 'both':
            h_stretch = 1.0 + random.uniform(-self.max_stretch, self.max_stretch)
        if stretch_type == 'v' or stretch_type == 'both':
            v_stretch = 1.0 + random.uniform(-self.max_stretch, self.max_stretch)
        
        # 应用拉伸变换
        if h_stretch != 1.0 or v_stretch != 1.0:
            width, height = img.size
            new_width = int(width * h_stretch)
            new_height = int(height * v_stretch)
            img = img.resize((new_width, new_height), Image.BILINEAR)
            
            # 调整回原始大小
            img = img.resize((width, height), Image.BILINEAR)
        
        # 随机旋转
        rotation_angle = random.uniform(-self.max_rotation, self.max_rotation)
        if rotation_angle != 0:
            img = img.rotate(rotation_angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
        
        return img

class PILGaussianBlur:
    """适用于PIL图像的高斯模糊"""
    def __init__(self, radius_min=0.1, radius_max=2.0):
        self.radius_min = radius_min
        self.radius_max = radius_max
        
    def __call__(self, img):
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius))

class PILRandomErasing:
    """适用于PIL图像的随机擦除"""
    def __init__(self, p=0.5, scale=(0.02, 0.3), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        
    def __call__(self, img):
        if random.random() > self.p:
            return img
            
        # 获取图像尺寸
        width, height = img.size
        
        # 随机选择擦除区域的大小
        area = width * height
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
        
        # 计算擦除区域尺寸
        h = int(round((target_area * aspect_ratio) ** 0.5))
        w = int(round((target_area / aspect_ratio) ** 0.5))
        
        if w < width and h < height:
            # 随机选择擦除位置
            x1 = random.randint(0, width - w)
            y1 = random.randint(0, height - h)
            
            # 创建擦除区域
            erase_box = (x1, y1, x1 + w, y1 + h)
            
            # 创建擦除图像
            erase_img = Image.new("L", (w, h), self.value)
            
            # 将擦除区域粘贴到原图上
            img.paste(erase_img, erase_box)
            
        return img

class PILThinning:
    """模拟细笔画效果"""
    def __call__(self, img):
        if random.random() < 0.3:  # 30%的概率应用细化
            img_np = np.array(img)
            thinned = cv2.ximgproc.thinning(
                img_np, 
                thinningType=cv2.ximgproc.THINNING_GUOHALL
            )
            return Image.fromarray(thinned)
        return img

class PILBlurVariation:
    """增加模糊多样性"""
    def __call__(self, img):
        radius = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # 更多模糊级别
        return img.filter(ImageFilter.GaussianBlur(radius))

class MNISTWithAugmentation(Dataset):
    def __init__(self, root, train=True, download=True, augment_ratio=0.5):
        self.base_dataset = torchvision.datasets.MNIST(
            root, train=train, download=download, transform=None)
        self.train = train
        self.augment_ratio = augment_ratio
        
        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 增强变换 - 全部在PIL图像上操作
        self.augment_transform = transforms.Compose([
            RandomStretchRotation(max_rotation=45, max_stretch=0.3),
            transforms.RandomAffine(
                degrees=0,  # 因为旋转已经在RandomStretchRotation中处理
                translate=(0.2, 0.2),
                scale=(0.7, 1.3),
                shear=20
            ),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            PILGaussianBlur(radius_min=0.1, radius_max=2.0),
            PILThinning(),          # 新增：细笔画模拟
            PILBlurVariation(),     # 新增：多样化模糊
            PILRandomErasing(p=0.5, scale=(0.02, 0.3), ratio=(0.3, 3.3)),  # 使用自定义的PIL随机擦除
            # 转换为张量并归一化
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            AddGaussianNoise(0., 0.1)
        ])
        
        # 创建增强样本
        self.augmented_samples = []
        if train:
            self._create_augmented_samples()
        
    def _create_augmented_samples(self):
        """创建增强样本并添加到数据集"""
        num_augmented = int(len(self.base_dataset) * self.augment_ratio)
        indices = random.sample(range(len(self.base_dataset)), num_augmented)
        
        for idx in indices:
            img, label = self.base_dataset[idx]
            # 应用增强变换
            augmented_img = self.augment_transform(img)
            self.augmented_samples.append((augmented_img, label))
    
    def __len__(self):
        return len(self.base_dataset) + len(self.augmented_samples)
        
    def __getitem__(self, idx):
        if idx < len(self.base_dataset):
            # 原始样本
            img, label = self.base_dataset[idx]
            return self.base_transform(img), label
        else:
            # 增强样本
            aug_idx = idx - len(self.base_dataset)
            img, label = self.augmented_samples[aug_idx]
            return img, label

# 焦点损失函数（提高对困难样本的关注）
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 数据集 - 使用增强比例为0.5 (额外50%增强样本)
    full_train_set = MNISTWithAugmentation('./data', train=True, download=True, augment_ratio=0.5)
    test_set = MNISTWithAugmentation('./data', train=False, download=True, augment_ratio=0.0)
    
    print(f"训练集大小: {len(full_train_set)} (原始:60000 + 增强:{len(full_train_set)-60000})")
    print(f"测试集大小: {len(test_set)}")

    # 分割数据集
    train_size = int(0.9 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = torch.utils.data.random_split(full_train_set, [train_size, val_size])

    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1024, num_workers=4, pin_memory=True)
    
    # 模型
    model = EnhancedCNN().to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    
    # 使用焦点损失
    criterion = FocalLoss(alpha=0.8, gamma=2.0)
    epochs = 100
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        # 训练
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 计算每个样本的损失
            with torch.no_grad():
                sample_loss = F.cross_entropy(output, target, reduction='none')
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 困难样本挖掘
            k = int(0.25 * len(sample_loss))
            hard_indices = sample_loss.topk(k).indices
            
            if k > 0:
                # 重新计算困难样本的损失
                optimizer.zero_grad()
                hard_output = model(data[hard_indices])
                hard_loss = criterion(hard_output, target[hard_indices])
                hard_loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs} [{batch_idx*len(data)}/{len(train_loader.dataset)}] '
                      f'Loss: {total_loss/(batch_idx+1):.6f}')
        
        # 验证
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                val_correct += model(data).argmax(1).eq(target).sum().item()
        
        val_acc = val_correct / len(val_loader.dataset)
        scheduler.step(val_acc)
        
        # 测试
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                test_correct += model(data).argmax(1).eq(target).sum().item()
        
        test_acc = test_correct / len(test_loader.dataset)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best test accuracy: {test_acc:.4f}')
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}: Val Acc {val_acc:.4f}, Test Acc {test_acc:.4f}, '
              f'Time {epoch_time:.2f}s, LR {optimizer.param_groups[0]["lr"]:.6f}')

    print(f"Best test accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    train()