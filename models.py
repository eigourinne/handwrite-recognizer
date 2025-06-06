# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1) * x

# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(concat)
        return self.sigmoid(att) * x

# 残差块（引入ResNet思想）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        return out

# 改进的CNN模型（引入ResNet和多尺度特征）
class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        # 残差块层
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # 注意力机制
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(128)
        self.sa2 = SpatialAttention()
        self.ca3 = ChannelAttention(256)
        self.sa3 = SpatialAttention()
        
        # 多尺度特征提取
        self.multiscale = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            ChannelAttention(32),
            SpatialAttention(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # 自适应池化
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 增强的分类器
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4 + 32*7*7, 1024),  # 合并多尺度特征
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 10)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)

    # 前向传播
    def forward(self, x):
        # 多尺度分支
        ms = self.multiscale(x)
        ms = ms.view(ms.size(0), -1)
        
        # 主分支
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 残差块 + 注意力
        x = self.layer1(x)
        x = self.ca1(x) + x
        x = self.sa1(x) + x
        
        x = self.layer2(x)
        x = self.ca2(x) + x
        x = self.sa2(x) + x
        
        x = self.layer3(x)
        x = self.ca3(x) + x
        x = self.sa3(x) + x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # 合并特征
        combined = torch.cat([x, ms], dim=1)
        return self.classifier(combined)