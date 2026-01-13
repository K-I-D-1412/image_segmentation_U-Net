import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block with two convolutional layers"""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # For matching dimensions

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class UNetResidual(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetResidual, self).__init__()

        # Encoder
        self.enc1 = ResidualBlock(in_channels, 8)
        self.enc2 = ResidualBlock(8, 16)
        self.enc3 = ResidualBlock(16, 32)
        self.enc4 = ResidualBlock(32, 64)
        self.enc5 = ResidualBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(32, 16)
        self.up1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(16, 8)

        # Final layer
        self.final = nn.Conv2d(8, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        # Decoder
        up4 = self.up4(enc5)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        # Final output
        return self.sigmoid(self.final(dec1))
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import JaccardIndex
import matplotlib.pyplot as plt

# 定义 Dice + BCE 组合损失
class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        bce = self.bce(inputs, targets)
        return bce + (1 - dice)

# 训练和验证函数
def train_model(model, train_loader, val_loader, device, num_epochs=50):
    model = model.to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
    iou_metric = JaccardIndex(task='binary', num_classes=2).to(device)

    train_losses = []
    val_losses = []
    val_ious = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                val_iou += iou_metric(preds, masks).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        scheduler.step(val_iou)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

    return train_losses, val_losses, val_ious

# 绘制 Loss 和 IoU 曲线
def plot_metrics(train_losses, val_losses, val_ious):
    epochs = len(train_losses)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), val_ious, label="Val IoU")
    plt.xlabel("Epochs")
    plt.ylabel("IoU")
    plt.legend()
    plt.title("IoU Curve")

    plt.tight_layout()
    plt.show()
