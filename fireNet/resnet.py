import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    """Basic ResNet block with two convolutional layers and a skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out


class ResNetUNet(nn.Module):
    """Lightweight ResNet-based UNet for fire mask prediction."""
    def __init__(self, in_channels=3, num_classes=1):
        super(ResNetUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder (ResNet blocks)
        self.encoder1 = ResNetBlock(in_channels, 64)
        self.encoder2 = ResNetBlock(64, 128, stride=2)
        self.encoder3 = ResNetBlock(128, 256, stride=2)
        self.encoder4 = ResNetBlock(256, 512, stride=2)

        # Decoder (Up-convolution and ResNet blocks)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ResNetBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ResNetBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ResNetBlock(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # Output: [B, 64, H, W]
        e2 = self.encoder2(e1)  # Output: [B, 128, H/2, W/2]
        e3 = self.encoder3(e2)  # Output: [B, 256, H/4, W/4]
        e4 = self.encoder4(e3)  # Output: [B, 512, H/8, W/8]

        # Decoder with skip connections
        d3 = self.upconv3(e4)  # Output: [B, 256, H/4, W/4]
        d3 = torch.cat([d3, e3], dim=1)  # Skip connection
        d3 = self.decoder3(d3)  # Output: [B, 256, H/4, W/4]

        d2 = self.upconv2(d3)  # Output: [B, 128, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        d2 = self.decoder2(d2)  # Output: [B, 128, H/2, W/2]

        d1 = self.upconv1(d2)  # Output: [B, 64, H, W]
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d1 = self.decoder1(d1)  # Output: [B, 64, H, W]

        # Final output
        out = self.final_conv(d1)  # Output: [B, num_classes, H, W]
        return torch.sigmoid(out)  # Use sigmoid for binary mask prediction