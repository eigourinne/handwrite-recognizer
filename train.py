# train.py
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models import EnhancedCNN
import time

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
        
    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

class MNISTWithAugmentation(Dataset):
    def __init__(self, root, train=True, download=True):
        self.base_dataset = torchvision.datasets.MNIST(
            root, train=train, download=download, transform=None)
        self.train = train
        
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(10, translate=(0.1,0.1), scale=(0.9,1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(3, sigma=(0.1,1.0)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02,0.2), ratio=(0.3,3.3)),
            transforms.Normalize((0.1307,), (0.3081,)),
            AddGaussianNoise(0., 0.05)
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.train:
            return self.train_transform(img), label
        return self.test_transform(img), label

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 数据集
    full_train_set = MNISTWithAugmentation('./data', train=True, download=True)
    test_set = MNISTWithAugmentation('./data', train=False, download=True)

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
    
    criterion = nn.CrossEntropyLoss()
    epochs = 100
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        # 训练
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
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