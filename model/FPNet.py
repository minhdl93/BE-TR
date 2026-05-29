"""
FPNet (Feature Pyramid Network) implementation for Salient Object Detection
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


class FPNet(nn.Module):
    """
    FPNet (Feature Pyramid Network) for Salient Object Detection
    Uses PVTv2 backbone with Feature Pyramid Network decoder
    """
    def __init__(self, channel=64):
        super(FPNet, self).__init__()
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

        # Lateral connections
        self.latlayer1 = BasicConv2d(channel, channel, 1)
        self.latlayer2 = BasicConv2d(channel, channel, 1)
        self.latlayer3 = BasicConv2d(channel, channel, 1)
        self.latlayer4 = BasicConv2d(channel, channel, 1)

        # FPN top-down pathway
        self.topdown1 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.topdown2 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.topdown3 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.topdown4 = convbnrelu(channel, channel, k=3, s=1, p=1)

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

        # Prediction heads
        self.pred_conv1 = nn.Conv2d(channel, 1, 1)
        self.pred_conv2 = nn.Conv2d(channel, 1, 1)
        self.pred_conv3 = nn.Conv2d(channel, 1, 1)
        self.pred_conv4 = nn.Conv2d(channel, 1, 1)
        self.pred_conv5 = nn.Conv2d(channel, 1, 1)

        # Refinement layers
        self.refine1 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.refine2 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.refine3 = convbnrelu(channel, channel, k=3, s=1, p=1)

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

        # FPN top-down pathway
        p4 = self.latlayer4(x4_t)
        p3 = self.topdown3(self.fusion1(torch.cat([
            F.interpolate(p4, size=x3_t.shape[2:], mode='bilinear', align_corners=False),
            self.latlayer3(x3_t)
        ], dim=1)))
        
        p2 = self.topdown2(self.fusion2(torch.cat([
            F.interpolate(p3, size=x2_t.shape[2:], mode='bilinear', align_corners=False),
            self.latlayer2(x2_t)
        ], dim=1)))
        
        p1 = self.topdown1(self.fusion3(torch.cat([
            F.interpolate(p2, size=x1_t.shape[2:], mode='bilinear', align_corners=False),
            self.latlayer1(x1_t)
        ], dim=1)))

        # Refinement
        r1 = self.refine1(p1)
        r2 = self.refine2(F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False))
        r3 = self.refine3(F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=False))

        # Multi-scale predictions
        pred1 = self.pred_conv1(r1)
        pred2 = self.pred_conv2(r2)
        pred3 = self.pred_conv3(r3)
        pred4 = self.pred_conv4(F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False))
        
        # Final fused prediction
        fused = r1 + r2 + r3
        pred5 = self.pred_conv5(fused)

        # Interpolate to original image size
        predictions = []
        for pred in [pred4, pred3, pred2, pred1, pred5]:
            pred = F.interpolate(pred, size=image_shape, mode='bilinear', align_corners=False)
            predictions.append(pred)

        return predictions

