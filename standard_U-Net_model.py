import torch
import torch.nn as nn
import os
import netron

# 设置环境变量，确保 Graphviz 可正常运行（如有需要）
os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                #nn.BatchNorm2d(out_c),   #数据量太小了，加BN反而会使准确度变很低，在0.6左右，高不了0.9
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                #nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        def up_conv(in_c, out_c):
            return nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)

        # Encoder
        self.encoder1 = conv_block(in_channels, 8)    # 输入: 3 -> 8
        self.encoder2 = conv_block(8, 16)            # 输入: 8 -> 16
        self.encoder3 = conv_block(16, 32)           # 输入: 16 -> 32
        self.encoder4 = conv_block(32, 64)           # 输入: 32 -> 64
        self.encoder5 = conv_block(64, 128)          # 输入: 64 -> 128

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv4 = up_conv(128, 128)
        self.decoder4 = conv_block(128 + 64, 64)    # 拼接后通道数 192

        self.upconv3 = up_conv(64, 64)
        self.decoder3 = conv_block(64 + 32, 32)      # 拼接后通道数 96

        self.upconv2 = up_conv(32, 32)
        self.decoder2 = conv_block(32 + 16, 16)      # 拼接后通道数 48

        self.upconv1 = up_conv(16, 16)
        self.decoder1 = conv_block(16 + 8, 8)       # 拼接后通道数 24

        # Final output layer
        self.final = nn.Conv2d(8, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))

        # Decoder
        up4 = self.upconv4(enc5)
        up4 = torch.cat([up4, enc4], dim=1)  # 拼接
        dec4 = self.decoder4(up4)

        up3 = self.upconv3(dec4)
        up3 = torch.cat([up3, enc3], dim=1)  # 拼接
        dec3 = self.decoder3(up3)

        up2 = self.upconv2(dec3)
        up2 = torch.cat([up2, enc2], dim=1)  # 拼接
        dec2 = self.decoder2(up2)

        up1 = self.upconv1(dec2)
        up1 = torch.cat([up1, enc1], dim=1)  # 拼接
        dec1 = self.decoder1(up1)

        # Final output
        output = self.final(dec1)
        return self.sigmoid(output)


# 实例化模型并放到 GPU
#model = UNet(in_channels=3, out_channels=1).to('cuda')

# 创建一个假输入
#dummy_input = torch.randn(1, 3, 128, 128).to('cuda')  # batch_size=1, 3通道, 高宽为128

# 保存模型的参数
#torch.save(model.state_dict(), "unet_model.pth")  # 保存仅包含参数的模型

# 保存完整模型
#torch.save(model, "unet_model_full.pth")  # 保存完整的模型

# 使用 Netron 可视化
#print("启动 Netron 以可视化模型...")
#netron.start("unet_model_full.pth")  # 打开完整模型进行可视化
