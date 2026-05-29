"""
Simplified Mask2Former implementation for Salient Object Detection
Compatible with the training pipeline in train_wpformer1.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pvtv2 import pvt_v2_b2
from model.position_encoding import PositionEmbeddingSine
from model.transformer import Transformer, SelfAttentionLayer, FFNLayer, MLP
from typing import Optional
from torch import Tensor


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


class Mask2Former(nn.Module):
    """
    Simplified Mask2Former for Salient Object Detection
    Based on the Mask2Former architecture with PVTv2 backbone
    """
    def __init__(self, channel=64, num_queries=100):
        super(Mask2Former, self).__init__()
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

        # Feature transformation layers
        self.Translayer1_1 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        # FPN-like decoder
        self.latlayer1 = BasicConv2d(channel, channel, 1)
        self.latlayer2 = BasicConv2d(channel, channel, 1)
        self.latlayer3 = BasicConv2d(channel, channel, 1)
        self.latlayer4 = BasicConv2d(channel, channel, 1)

        self.outconv1 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.outconv2 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.outconv3 = convbnrelu(channel, channel, k=3, s=1, p=1)
        self.outconv4 = convbnrelu(channel, channel, k=3, s=1, p=1)

        # Upsample and fusion
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

        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, channel)
        
        # Positional encoding
        N_steps = channel // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # Transformer decoder
        self.num_heads = 8
        self.num_feature_levels = 3
        
        # Transformer for mask prediction
        self.transformer = Transformer(
            d_model=channel,
            dropout=0.1,
            nhead=self.num_heads,
            dim_feedforward=2048,
            num_encoder_layers=0,
            num_decoder_layers=6,
            normalize_before=False,
            return_intermediate_dec=True,
        )
        
        # Input projection for transformer
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            self.input_proj.append(convbnrelu(channel, channel, k=1, s=1, p=0))
        
        self.level_embed = nn.Embedding(self.num_feature_levels, channel)
        
        # Prediction heads
        self.class_embed = nn.Linear(channel, 1)
        self.mask_embed = MLP(channel, channel, channel, 3)
        self.mask_features = convbnrelu(channel, channel, k=3, s=1, p=1)
        
        # Final prediction layers
        self.pred_conv1 = nn.Conv2d(channel, 1, 1)
        self.pred_conv2 = nn.Conv2d(channel, 1, 1)
        self.pred_conv3 = nn.Conv2d(channel, 1, 1)
        self.pred_conv4 = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        image_shape = x.size()[2:]
        bs = x.size()[0]
        
        # Backbone features
        pvt = self.backbone(x)
        x1, x2, x3, x4 = pvt[0], pvt[1], pvt[2], pvt[3]

        # Transform features
        x1_t = self.Translayer1_1(x1)
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)

        # FPN decoder
        d3 = self.outconv3(self.fusion1(torch.cat([
            F.interpolate(self.latlayer3(x4_t), size=x3_t.shape[2:], mode='bilinear', align_corners=False),
            x3_t
        ], dim=1)))
        
        d2 = self.outconv2(self.fusion2(torch.cat([
            F.interpolate(d3, size=x2_t.shape[2:], mode='bilinear', align_corners=False),
            x2_t
        ], dim=1)))
        
        d1 = self.outconv1(self.fusion3(torch.cat([
            F.interpolate(d2, size=x1_t.shape[2:], mode='bilinear', align_corners=False),
            x1_t
        ], dim=1)))

        # Multi-scale features for transformer
        features = [x4_t, d3, d2]
        
        # Prepare transformer inputs
        src = []
        pos = []
        for i in range(self.num_feature_levels):
            feat = features[i]
            pos_embed = self.pe_layer(feat, None).flatten(2)
            src_feat = self.input_proj[i](feat).flatten(2) + self.level_embed.weight[i][None, :, None]
            
            pos.append(pos_embed.permute(2, 0, 1))
            src.append(src_feat.permute(2, 0, 1))

        # Query embeddings (pass raw weight, transformer will handle unsqueeze/repeat)
        mask_features = self.mask_features(d1)

        # Get query embeddings and ensure it's 2D [num_queries, channel]
        # This is important for DataParallel compatibility
        query_embed = self.query_embed.weight
        if query_embed.dim() != 2:
            # If somehow it's not 2D, reshape it
            query_embed = query_embed.view(-1, query_embed.size(-1))
        
        # Transformer decoder
        hs, _ = self.transformer(
            self.pe_layer(mask_features), None, query_embed,
            self.input_proj[0](mask_features), None
        )

        # Generate predictions from transformer outputs
        predictions = []
        
        # Multi-scale predictions
        for i, h in enumerate(hs):
            decoder_output = h.transpose(0, 1)  # [bs, num_queries, channel]
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            
            # Aggregate queries
            pred = torch.einsum("bqc,bqhw->bchw", outputs_class, outputs_mask)
            pred = F.interpolate(pred, size=image_shape, mode='bilinear', align_corners=False)
            predictions.append(pred)

        # Additional direct predictions for multi-scale supervision
        pred1 = self.pred_conv1(d1)
        pred2 = self.pred_conv2(d2)
        pred3 = self.pred_conv3(d3)
        pred4 = self.pred_conv4(x4_t)
        
        pred1 = F.interpolate(pred1, size=image_shape, mode='bilinear', align_corners=False)
        pred2 = F.interpolate(pred2, size=image_shape, mode='bilinear', align_corners=False)
        pred3 = F.interpolate(pred3, size=image_shape, mode='bilinear', align_corners=False)
        pred4 = F.interpolate(pred4, size=image_shape, mode='bilinear', align_corners=False)
        
        # Return list of predictions for multi-scale supervision
        return [pred4, pred3, pred2, pred1] + predictions[:2]  # Return 6 predictions total

