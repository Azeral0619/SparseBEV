import numpy as np
import torch
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner.base_module import BaseModule, Sequential
from torch import nn
from torch.cuda.amp.autocast_mode import autocast


@PLUGIN_LAYERS.register_module()
class DepthReweightModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        min_depth=1,
        max_depth=56,
        depth_interval=5,
        ffn_layers=2,
    ):
        super(DepthReweightModule, self).__init__()
        self.embed_dims = embed_dims
        self.min_depth = min_depth
        self.depth_interval = depth_interval
        self.depths = np.arange(min_depth, max_depth + 1e-5, depth_interval)
        self.max_depth = max(self.depths)

        layers = []
        for i in range(ffn_layers):
            layers.append(
                FFN(
                    embed_dims=embed_dims,
                    feedforward_channels=embed_dims,
                    num_fcs=2,
                    act_cfg=dict(type="ReLU", inplace=True),
                    dropout=0.0,
                    add_residual=True,
                )
            )
        layers.append(nn.Linear(embed_dims, len(self.depths)))
        self.depth_fc = Sequential(*layers)

    def forward(self, features, points_3d, output_conf=False):
        reference_depths = torch.norm(points_3d[..., :2], dim=-1, p=2, keepdim=True)
        reference_depths = torch.clip(
            reference_depths,
            max=self.max_depth - 1e-5,
            min=self.min_depth + 1e-5,
        )
        weights = (
            1
            - torch.abs(reference_depths - points_3d.new_tensor(self.depths))
            / self.depth_interval
        )

        top2 = weights.topk(2, dim=-1)[0]
        weights = torch.where(
            weights >= top2[..., 1:2], weights, weights.new_tensor(0.0)
        )
        scale = torch.pow(top2[..., 0:1], 2) + torch.pow(top2[..., 1:2], 2)
        confidence = self.depth_fc(features).softmax(dim=-1)
        confidence = torch.sum(weights * confidence, dim=-1, keepdim=True)
        confidence = confidence / scale

        if output_conf:
            return confidence
        return features * confidence


@PLUGIN_LAYERS.register_module()
class DenseDepthNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = (depth.T * focal / self.equal_focal).T
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(gt > 0.0, torch.logical_not(torch.isnan(pred)))
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = error / max(1.0, len(gt) * len(depth_preds)) * self.loss_weight
            loss = loss + _loss
        return loss
