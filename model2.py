import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        # 通道数调整
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm2d(F_int)
        )
        # 激活和卷积
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 调整 g 的尺寸以匹配 x
        g1 = self.W_g(g)  # [B, F_int, H', W']
        x1 = self.W_x(x)  # [B, F_int, H, W]

        # 如果 g1 的尺寸与 x1 不一致，则进行上采样
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)

        # 相加并激活
        psi = self.relu(g1 + x1)  # [B, F_int, H, W]
        psi = self.psi(psi)  # [B, 1, H, W]

        # 注意力加权
        return x * psi



class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # 编码器部分
        self.encoder1 = self.conv_block(in_channels, 8)
        self.encoder2 = self.conv_block(8, 16)
        self.encoder3 = self.conv_block(16, 32)
        self.encoder4 = self.conv_block(32, 64)
        self.encoder5 = self.conv_block(64, 128)  # 中间部分

        # 注意力模块
        self.att4 = AttentionBlock(F_g=128, F_l=64, F_int=64)  # F_g 是解码器的输入通道，F_l 是编码器的输出通道
        self.att3 = AttentionBlock(F_g=64, F_l=32, F_int=32)
        self.att2 = AttentionBlock(F_g=32, F_l=16, F_int=16)
        self.att1 = AttentionBlock(F_g=16, F_l=8, F_int=8)

        # 解码器部分
        self.decoder4 = self.conv_block(128 + 64, 64)
        self.decoder3 = self.conv_block(64 + 32, 32)
        self.decoder2 = self.conv_block(32 + 16, 16)
        self.decoder1 = self.conv_block(16 + 8, 8)

        # 最终卷积层
        self.final_conv = nn.Conv2d(8, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        """构造一个卷积块：两层卷积+BatchNorm+ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码部分
        enc1 = self.encoder1(x)  # [B, 8, H, W]
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))  # [B, 16, H/2, W/2]
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))  # [B, 32, H/4, W/4]
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))  # [B, 64, H/8, W/8]
        enc5 = self.encoder5(F.max_pool2d(enc4, 2))  # [B, 128, H/16, W/16]

        # 解码部分 + 注意力机制
        _enc4 = self.att4(enc5, enc4)  # 使用注意力模块对特征进行加权
        dec4 = self.decoder4(
            torch.cat([F.interpolate(enc5, scale_factor=2, mode='bilinear', align_corners=True), _enc4],
                      dim=1))  # [B, 64, H/8, W/8]

        _enc3 = self.att3(dec4, enc3)
        dec3 = self.decoder3(
            torch.cat([F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), _enc3],
                      dim=1))  # [B, 32, H/4, W/4]

        _enc2 = self.att2(dec3, enc2)
        dec2 = self.decoder2(
            torch.cat([F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), _enc2],
                      dim=1))  # [B, 16, H/2, W/2]

        _enc1 = self.att1(dec2, enc1)
        dec1 = self.decoder1(
            torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), _enc1],
                      dim=1))  # [B, 8, H, W]

        # 最终输出
        out = self.final_conv(dec1)  # [B, 1, H, W]
        return self.sigmoid(out)

# 实例化U-Net模型
model = UNet(in_channels=3, out_channels=1)

# 创建一个样本输入张量，确保它与模型的输入大小匹配
x = torch.randn(1, 3, 128, 128)  # batch size=1, 3通道，128x128的图像尺寸

# 打印模型概况
print(model)

# 运行模型并查看输出形状
output = model(x)
print(f"Output shape: {output.shape}")