import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)

# ============================================================
# Basic convolution block
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Binary segmentation U-Net.

    Input:
        [B, 1, H, W]

    Output:
        [B, 1, H, W] logits

    Geometry/Tissue の両方で foreground = abnormal として学習する。
    Healthy は foreground を持たない空マスクを使用する。
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.up1 = Up(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
        )
        self.up2 = Up(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
        )
        self.up3 = Up(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
        )
        self.up4 = Up(
            base_channels * 2,
            base_channels,
            base_channels,
        )

        self.outc = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

# ============================================================
# Decoder block
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_channels
            + skip_channels,

            out_channels,
        )

    def forward(
        self,
        x,
        skip,
    ):

        # =========================
        # Upsampling
        # =========================

        x = self.up(
            x
        )

        # =========================
        # Size adjustment
        # =========================

        if (
            x.shape[-2:]
            != skip.shape[-2:]
        ):

            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # =========================
        # Skip connection
        # =========================

        x = torch.cat(
            [
                x,
                skip,
            ],
            dim=1,
        )

        x = self.conv(
            x
        )

        return x

# ============================================================
# ResNet U-Net
# ============================================================

class ResNetUNet(nn.Module):

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        in_channels=1,
        out_channels=1,
    ):

        super().__init__()

        # ====================================================
        # Select backbone
        # ====================================================

        if backbone == "resnet18":

            weights = (
                ResNet18_Weights.DEFAULT
                if pretrained
                else None
            )

            encoder = resnet18(
                weights=weights
            )

            encoder_channels = [
                64,
                64,
                128,
                256,
                512,
            ]

        elif backbone == "resnet34":

            weights = (
                ResNet34_Weights.DEFAULT
                if pretrained
                else None
            )

            encoder = resnet34(
                weights=weights
            )

            encoder_channels = [
                64,
                64,
                128,
                256,
                512,
            ]

        elif backbone == "resnet50":

            weights = (
                ResNet50_Weights.DEFAULT
                if pretrained
                else None
            )

            encoder = resnet50(
                weights=weights
            )

            encoder_channels = [
                64,
                256,
                512,
                1024,
                2048,
            ]

        else:

            raise ValueError(
                f"Unsupported backbone: "
                f"{backbone}"
            )

        # ====================================================
        # Convert first Conv from RGB -> grayscale
        # ====================================================

        if in_channels == 1:

            original_conv = (
                encoder.conv1
            )

            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

            # ================================================
            # ImageNet pretrained weight:
            #
            # RGB:
            # [64, 3, 7, 7]
            #
            # ->
            #
            # grayscale:
            # [64, 1, 7, 7]
            # ================================================

            if pretrained:

                with torch.no_grad():

                    new_conv.weight.copy_(
                        original_conv.weight.mean(
                            dim=1,
                            keepdim=True,
                        )
                    )

            encoder.conv1 = (
                new_conv
            )

        elif in_channels != 3:

            raise ValueError(
                "in_channels must be "
                "1 or 3."
            )

        # ====================================================
        # Encoder
        # ====================================================

        self.conv1 = (
            encoder.conv1
        )

        self.bn1 = (
            encoder.bn1
        )

        self.relu = (
            encoder.relu
        )

        self.maxpool = (
            encoder.maxpool
        )

        self.layer1 = (
            encoder.layer1
        )

        self.layer2 = (
            encoder.layer2
        )

        self.layer3 = (
            encoder.layer3
        )

        self.layer4 = (
            encoder.layer4
        )

        # ====================================================
        # Decoder
        # ====================================================

        c0, c1, c2, c3, c4 = (
            encoder_channels
        )

        self.decoder4 = DecoderBlock(
            in_channels=c4,
            skip_channels=c3,
            out_channels=256,
        )

        self.decoder3 = DecoderBlock(
            in_channels=256,
            skip_channels=c2,
            out_channels=128,
        )

        self.decoder2 = DecoderBlock(
            in_channels=128,
            skip_channels=c1,
            out_channels=64,
        )

        self.decoder1 = DecoderBlock(
            in_channels=64,
            skip_channels=c0,
            out_channels=64,
        )

        # ====================================================
        # Final upsample
        #
        # H/2 -> H
        # ====================================================

        self.final_up = (
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2,
            )
        )

        self.final_conv = (
            nn.Sequential(

                DoubleConv(
                    32,
                    32,
                ),

                nn.Conv2d(
                    32,
                    out_channels,
                    kernel_size=1,
                ),
            )
        )

    def forward(
        self,
        x,
    ):

        # ====================================================
        # Encoder
        # ====================================================

        # Input:
        #
        # [B, 1, H, W]

        x0 = self.conv1(
            x
        )

        x0 = self.bn1(
            x0
        )

        x0 = self.relu(
            x0
        )

        # x0:
        # H / 2

        x1 = self.maxpool(
            x0
        )

        x1 = self.layer1(
            x1
        )

        # x1:
        # H / 4

        x2 = self.layer2(
            x1
        )

        # x2:
        # H / 8

        x3 = self.layer3(
            x2
        )

        # x3:
        # H / 16

        x4 = self.layer4(
            x3
        )

        # x4:
        # H / 32

        # ====================================================
        # Decoder
        # ====================================================

        d4 = self.decoder4(
            x4,
            x3,
        )

        # H / 16

        d3 = self.decoder3(
            d4,
            x2,
        )

        # H / 8

        d2 = self.decoder2(
            d3,
            x1,
        )

        # H / 4

        d1 = self.decoder1(
            d2,
            x0,
        )

        # H / 2

        # ====================================================
        # Full resolution
        # ====================================================

        out = self.final_up(
            d1
        )

        out = self.final_conv(
            out
        )

        # Output:
        #
        # [B, 1, H, W]

        return out


# ============================================================
# Model builder
# ============================================================

def build_model(
    model_cfg,
):

    model_name = model_cfg[
        "name"
    ].lower()

    if model_name == "unet_resnet":

        model = ResNetUNet(

            backbone=
                model_cfg.get(
                    "backbone",
                    "resnet34",
                ),

            pretrained=
                model_cfg.get(
                    "pretrained",
                    True,
                ),

            in_channels=
                int(
                    model_cfg.get(
                        "in_channels",
                        1,
                    )
                ),

            out_channels=
                int(
                    model_cfg.get(
                        "out_channels",
                        1,
                    )
                ),
        )

        return model
    elif model_name == "unet": 
        model = UNet(
            in_channels=int(
                model_cfg["in_channels"]
            ),
            out_channels=int(
                model_cfg["out_channels"]
            ),
            base_channels=int(
                model_cfg["base_channels"]
            ),
        )
        return model

    raise ValueError(
        f"Unknown model: "
        f"{model_name}"
    )