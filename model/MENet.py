"""
MENet (Multi-scale Encoder Network) implementation for Salient Object Detection
Compatible with the training pipeline in train_wpformer1.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pvtv2 import pvt_v2_b2


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class MultiScaleEncoder(nn.Module):
    """Multi-scale encoder block"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleEncoder, self).__init__()
        # Multi-scale convolutions
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x7 = self.conv7x7(x)
        
        out = torch.cat([x1, x3, x5, x7], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class MENet(nn.Module):
    """
    MENet (Multi-scale Encoder Network) for Salient Object Detection
    Uses PVTv2 backbone with multi-scale encoder blocks
    """
    def __init__(self, channel=64):
        super(MENet, self).__init__()
        # Backbone
        self.backbone = pvt_v2_b2()
        path = 'model/pvt_v2_b2.pth'
        try:
            save_model = torch.load(path, map_location='cpu')
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        except:
            print(f"Warning: Could not load pretrained weights from {path}")

        # Feature transformation
        self.Translayer1_1 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        # Multi-scale encoders
        self.mse1 = MultiScaleEncoder(channel, channel)
        self.mse2 = MultiScaleEncoder(channel, channel)
        self.mse3 = MultiScaleEncoder(channel, channel)
        self.mse4 = MultiScaleEncoder(channel, channel)

        # Decoder layers
        self.decoder1 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.decoder2 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.decoder3 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.decoder4 = convbnrelu(channel, channel, k=3, s=1, p=1)

        # Feature fusion
        self.fusion1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.fusion3 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel, 1),
            nn.Sigmoid()
        )

        # Prediction heads
        self.pred_conv1 = nn.Conv2d(channel, 1, 1)
        self.pred_conv2 = nn.Conv2d(channel, 1, 1)
        self.pred_conv3 = nn.Conv2d(channel, 1, 1)
        self.pred_conv4 = nn.Conv2d(channel, 1, 1)
        self.pred_conv5 = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        image_shape = x.size()[2:]
        
        # Backbone features
        pvt = self.backbone(x)
        x1, x2, x3, x4 = pvt[0], pvt[1], pvt[2], pvt[3]

        # Transform features
        x1_t = self.Translayer1_1(x1)
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)

        # Multi-scale encoding
        e4 = self.mse4(x4_t)
        e3 = self.mse3(x3_t)
        e2 = self.mse2(x2_t)
        e1 = self.mse1(x1_t)

        # Top-down decoder with fusion
        d3 = self.decoder3(self.fusion1(torch.cat([
            F.interpolate(e4, size=e3.shape[2:], mode='bilinear', align_corners=False),
            e3
        ], dim=1)))
        
        d2 = self.decoder2(self.fusion2(torch.cat([
            F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False),
            e2
        ], dim=1)))
        
        d1 = self.decoder1(self.fusion3(torch.cat([
            F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False),
            e1
        ], dim=1)))

        # Apply attention
        att = self.attention(d1)
        d1_att = d1 * att

        # Multi-scale predictions
        pred1 = self.pred_conv1(d1_att)
        pred2 = self.pred_conv2(d2)
        pred3 = self.pred_conv3(d3)
        pred4 = self.pred_conv4(e4)
        
        # Final fused prediction
        fused = d1_att + F.interpolate(d2, size=d1_att.shape[2:], mode='bilinear', align_corners=False)
        pred5 = self.pred_conv5(fused)

        # Interpolate to original image size
        predictions = []
        for pred in [pred4, pred3, pred2, pred1, pred5]:
            pred = F.interpolate(pred, size=image_shape, mode='bilinear', align_corners=False)
            predictions.append(pred)

        return predictions

