import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import bias_init_with_prob
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmdet.models.utils.builder import TRANSFORMER
from .bbox.utils import decode_bbox
from .utils import inverse_sigmoid, DUMP
from .sparsebev_sampling import sampling_4d, make_sample_points
from .checkpoint import checkpoint as cp
from .csrc.wrapper import MSMV_CUDA


@TRANSFORMER.register_module()
class SparseBEVTransformer(BaseModule):
    def __init__(
        self,
        embed_dims,
        num_frames=8,
        num_points=4,
        num_layers=6,
        num_levels=4,
        num_classes=10,
        num_views=6,
        code_size=10,
        pc_range=[],
        init_cfg=None,
    ):
        assert init_cfg is None, (
            "To prevent abnormal initialization "
            "behavior, init_cfg is not allowed to be set"
        )
        super(SparseBEVTransformer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range

        self.decoder = SparseBEVTransformerDecoder(
            embed_dims,
            num_frames,
            num_points,
            num_layers,
            num_levels,
            num_classes,
            num_views,
            code_size,
            pc_range=pc_range,
        )

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_bbox, query_feat, mlvl_feats, attn_mask, img_metas):
        cls_scores, bbox_preds = self.decoder(
            query_bbox, query_feat, mlvl_feats, attn_mask, img_metas
        )

        cls_scores = torch.nan_to_num(cls_scores)
        bbox_preds = torch.nan_to_num(bbox_preds)

        return cls_scores, bbox_preds


class SparseBEVTransformerDecoder(BaseModule):
    def __init__(
        self,
        embed_dims,
        num_frames=8,
        num_points=4,
        num_layers=6,
        num_levels=4,
        num_classes=10,
        num_views=6,
        code_size=10,
        pc_range=[],
        init_cfg=None,
    ):
        super(SparseBEVTransformerDecoder, self).__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.num_views = num_views

        # params are shared across all decoder layers
        self.decoder_layer = SparseBEVTransformerDecoderLayer(
            embed_dims,
            num_frames,
            num_points,
            num_levels,
            num_classes,
            num_views,
            code_size,
            pc_range=pc_range,
        )

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layer.init_weights()

    def forward(self, query_bbox, query_feat, mlvl_feats, attn_mask, img_metas):
        cls_scores, bbox_preds = [], []

        # calculate time difference according to timestamps
        timestamps = np.array([m["img_timestamp"] for m in img_metas], dtype=np.float64)
        timestamps = np.reshape(timestamps, [query_bbox.shape[0], -1, self.num_views])
        time_diff = timestamps[:, :1, :] - timestamps
        time_diff = np.mean(time_diff, axis=-1).astype(np.float32)  # [B, F]
        time_diff = torch.from_numpy(time_diff).to(query_bbox.device)  # [B, F]
        img_metas[0]["time_diff"] = time_diff

        # organize projections matrix and copy to CUDA
        lidar2img = np.asarray([m["lidar2img"] for m in img_metas]).astype(np.float32)
        lidar2img = torch.from_numpy(lidar2img).to(query_bbox.device)  # [B, N, 4, 4]
        img_metas[0]["lidar2img"] = lidar2img

        # 重新组织特征，以便于后续的采样
        # group image features in advance for sampling, see `sampling_4d` for more details
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
            N, T, G, C = (
                self.num_views,
                TN // self.num_views,
                4,
                GC // 4,
            )  # TODO: change num of views
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:  # Our CUDA operator requires channel_last
                # [B, T, G, N, H, W, C]
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)
                feat = feat.reshape(B * T * G, N, H, W, C)
            else:  # Torch's grid_sample requires channel_first
                # [B, T, G, C, N, H, W]
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)
                feat = feat.reshape(B * T * G, C, N, H, W)

            mlvl_feats[lvl] = feat.contiguous()

        # 进行六层解码
        for i in range(self.num_layers):
            DUMP.stage_count = i

            query_feat, cls_score, bbox_pred = self.decoder_layer(
                query_bbox, query_feat, mlvl_feats, attn_mask, img_metas
            )

            # 使用预测的结果更新查询Pillar
            query_bbox = bbox_pred.clone().detach()

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        cls_scores = torch.stack(cls_scores)
        bbox_preds = torch.stack(bbox_preds)

        return cls_scores, bbox_preds


class SparseBEVTransformerDecoderLayer(BaseModule):
    def __init__(
        self,
        embed_dims,
        num_frames=8,
        num_points=4,
        num_levels=4,
        num_classes=10,
        num_views=6,
        code_size=10,
        num_cls_fcs=2,
        num_reg_fcs=2,
        pc_range=[],
        init_cfg=None,
    ):
        super(SparseBEVTransformerDecoderLayer, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.code_size = code_size
        self.pc_range = pc_range

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = SparseBEVSelfAttention(
            embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range
        )
        self.sampling = SparseBEVSampling(
            embed_dims,
            num_frames=num_frames,
            num_groups=4,
            num_points=num_points,
            num_levels=num_levels,
            num_views=num_views,
            pc_range=pc_range,
        )
        self.mixing = AdaptiveMixing(
            in_dim=embed_dims,
            in_points=num_points * num_frames,
            n_groups=4,
            out_points=128,
        )
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        self.reg_branch = nn.Sequential(*reg_branch)

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)  # type: ignore

    def refine_bbox(self, bbox_proposal, bbox_delta):
        # 把预测的位置增量加到查询pillar上，得到新的位置
        xyz = inverse_sigmoid(bbox_proposal[..., 0:3])
        xyz_delta = bbox_delta[..., 0:3]
        xyz_new = torch.sigmoid(xyz_delta + xyz)

        # 把更新后的位置与预测尺寸、方向、速度拼接起来，得到新的预测
        return torch.cat([xyz_new, bbox_delta[..., 3:]], dim=-1)

    # @info SparseBEV的核心代码
    def forward(self, query_bbox, query_feat, mlvl_feats, attn_mask, img_metas):
        """
        query_bbox: [B, Q, 10] [cx, cy, cz, w, h, d, rot.sin, rot.cos, vx, vy]
        """

        # 通过两层全连接网络进行位置编码，输入是查询pillar的坐标，然后把位置编码加到查询特征上
        query_pos = self.position_encoder(query_bbox[..., :3])
        query_feat = query_feat + query_pos

        # 解码器第一层：尺度自适应的多头自注意力
        query_feat = self.norm1(self.self_attn(query_bbox, query_feat, attn_mask))
        # TODO: Pre-LN or Post-LN
        # Post-LN
        # query_feat = self.norm1(query_feat + self.self_attn(query_bbox, query_feat, attn_mask))
        # Pre-LN
        # query_feat = self.self_attn(query_bbox, self.norm1(query_feat), attn_mask) + query_feat

        # 解码器第二层：从多尺度特征金字塔中采样
        sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, img_metas)

        # 解码器第三层：融合采样特征和查询特征
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        # TODO: Pre-LN or Post-LN
        # Post-LN
        # query_feat = self.norm2(query_feat + self.mixing(sampled_feat, query_feat))
        # Pre-LN
        # query_feat = self.mixing(sampled_feat, self.norm2(query_feat)) + query_feat

        # 解码器第四层：FFN网络
        query_feat = self.norm3(self.ffn(query_feat))
        # TODO: Pre-LN or Post-LN
        # Post-LN
        # query_feat = self.norm3(query_feat + self.ffn(query_feat))
        # Pre-LN
        # query_feat = self.ffn(self.norm3(query_feat)) + query_feat

        # 通过两层全连接网络，获得分类分数和回归目标
        cls_score = self.cls_branch(query_feat)  # [B, Q, num_classes]
        # 回归的结果是完整的pillar，包括位置、尺寸、方向、速度，但是位置是相对位置
        bbox_pred = self.reg_branch(query_feat)  # [B, Q, code_size]
        # 使用查询pillar的坐标+预测的相对位置，获得绝对位置
        bbox_pred = self.refine_bbox(query_bbox, bbox_pred)

        # 更新回归目标的速度信息
        # calculate absolute velocity according to time difference
        time_diff = img_metas[0]["time_diff"]  # [B, F]
        if time_diff.shape[1] > 1:
            time_diff = time_diff.clone()
            time_diff[time_diff < 1e-5] = 1.0
            bbox_pred[..., 8:] = bbox_pred[..., 8:] / time_diff[:, 1:2, None]

        if DUMP.enabled:
            query_bbox_dec = decode_bbox(query_bbox, self.pc_range)
            bbox_pred_dec = decode_bbox(bbox_pred, self.pc_range)
            cls_score_sig = torch.sigmoid(cls_score)
            torch.save(
                query_bbox_dec.cpu(),
                "{}/query_bbox_stage{}.pth".format(DUMP.out_dir, DUMP.stage_count),
            )
            torch.save(
                bbox_pred_dec.cpu(),
                "{}/bbox_pred_stage{}.pth".format(DUMP.out_dir, DUMP.stage_count),
            )
            torch.save(
                cls_score_sig.cpu(),
                "{}/cls_score_stage{}.pth".format(DUMP.out_dir, DUMP.stage_count),
            )

        return query_feat, cls_score, bbox_pred


class SparseBEVSelfAttention(BaseModule):
    """Scale-adaptive Self Attention"""

    def __init__(
        self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=[], init_cfg=None
    ):
        super().__init__(init_cfg)
        self.pc_range = pc_range

        self.attention = MultiheadAttention(
            embed_dims, num_heads, dropout, batch_first=True
        )
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_bbox, query_feat, pre_attn_mask):
        """
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        """

        # 计算所有查询pillar的中心点距离D，形状是[B, Q, Q]
        dist = self.calc_bbox_dists(query_bbox)
        # 通过对查询特征的线性变换，获得每个查询pillar的8个τ值，对应8个检测头，形状是[B, Q, 8]
        # τ控制感受野，当τ=0时退化为全局感受野的标准自注意力机制；
        # 随着τ的增加，远处查询的注意力权重变小，感受野变小
        tau = self.gen_tau(query_feat)  # [B, Q, 8]

        if DUMP.enabled:
            torch.save(
                tau.cpu(),
                "{}/sasa_tau_stage{}.pth".format(DUMP.out_dir, DUMP.stage_count),
            )

        # 计算中心点距离D和τ的乘积，形状是[B, 8, Q, Q]
        # dist*tau的效果是：
        # 1. dist张量被复制8份，每份对一个检测头，形状是[Q, Q]（忽略B）
        # 2. tau张量按检测头被分成8个向量，每个向量长度是Q（忽略B）
        # 3. dist的每一份分别乘以tau的每一个向量
        # 4. 然后8份张量按照第一个维度拼接起来
        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]

        if pre_attn_mask is not None:  # for query denoising
            attn_mask[:, :, pre_attn_mask] = float("-inf")

        # 把8个检测头的注意力权重拼接起来，形状是[Bx8, Q, Q]
        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]

        # 把attn_mask给到多头自注意力，实现尺度自适应的多头自注意力
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_bbox, query_feat, pre_attn_mask):
        if self.training and query_feat.requires_grad:
            return cp(
                self.inner_forward,
                query_bbox,
                query_feat,
                pre_attn_mask,
                use_reentrant=False,
            )
        else:
            return self.inner_forward(query_bbox, query_feat, pre_attn_mask)

    @torch.no_grad()
    def calc_bbox_dists(self, bboxes):
        # 取得所有查询pillar的中心点坐标（x, y），形状是[B, Q, 2]
        centers = decode_bbox(bboxes, self.pc_range)[..., :2]  # [B, Q, 2]

        # 计算所有查询pillar的相互中心点距离，形状是[B, Q, Q]
        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(
                centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1
            )
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)  # [B, Q, Q]
        dist = -dist

        return dist


class SparseBEVSampling(BaseModule):
    """Adaptive Spatio-temporal Sampling"""

    def __init__(
        self,
        embed_dims=256,
        num_frames=4,
        num_groups=4,
        num_points=8,
        num_levels=4,
        num_views=6,
        pc_range=[],
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range
        self.num_views = num_views

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_bbox, query_feat, mlvl_feats, img_metas):
        """
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        """
        B, Q = query_bbox.shape[:2]
        image_h, image_w, _ = img_metas[0]["img_shape"][0]

        # 通过对查询pillar的线性变换，获得采样点的偏移量，形状是[B, Q, G*P*3]
        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(
            B, Q, self.num_groups * self.num_points, 3
        )
        # 基于查询pillar的中心点坐标，计算采样点的坐标，形状是[B, Q, G*P, 3]
        sampling_points = make_sample_points(
            query_bbox, sampling_offset, self.pc_range
        )  # [B, Q, GP, 3]
        sampling_points = sampling_points.reshape(
            B, Q, 1, self.num_groups, self.num_points, 3
        )
        sampling_points = sampling_points.expand(
            B, Q, self.num_frames, self.num_groups, self.num_points, 3
        )

        # 基于目标的速度对采样位置进行修正
        # warp sample points based on velocity
        # 根据时差和速度计算位移，然后对采样点进行修正
        time_diff = img_metas[0]["time_diff"]  # [B, F]
        time_diff = time_diff[:, None, :, None]  # [B, 1, F, 1]
        vel = query_bbox[..., 8:].detach()  # [B, Q, 2]
        vel = vel[:, :, None, :]  # [B, Q, 1, 2]
        dist = vel * time_diff  # [B, Q, F, 2]
        dist = dist[:, :, :, None, None, :]  # [B, Q, F, 1, 1, 2]
        # 修正采样点的x和y坐标
        sampling_points = torch.cat(
            [sampling_points[..., 0:2] - dist, sampling_points[..., 2:3]], dim=-1
        )

        # 通过对查询pillar的线性变换，获得每个采样点的权重，形状是[B, Q, G, T, P, L]
        # scale weights
        scale_weights = self.scale_weights(query_feat).view(
            B, Q, self.num_groups, 1, self.num_points, self.num_levels
        )
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(
            B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels
        )

        # 进行采样
        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            img_metas[0]["lidar2img"],
            image_h,
            image_w,
            self.num_views,
        )  # [B, Q, G, FP, C]

        return sampled_feats

    def forward(self, query_bbox, query_feat, mlvl_feats, img_metas):
        if self.training and query_feat.requires_grad:
            return cp(
                self.inner_forward,
                query_bbox,
                query_feat,
                mlvl_feats,
                img_metas,
                use_reentrant=False,
            )
        else:
            return self.inner_forward(query_bbox, query_feat, mlvl_feats, img_metas)


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""

    def __init__(
        self,
        in_dim,
        in_points,
        n_groups=1,
        query_dim=None,
        out_dim=None,
        out_points=None,
    ):
        super(AdaptiveMixing, self).__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(
            self.query_dim, self.n_groups * self.total_parameters
        )
        self.out_proj = nn.Linear(
            self.eff_out_dim * self.out_points * self.n_groups, self.query_dim
        )
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        # 通过对查询的线性变换，获得混合参数
        """generate mixing parameters"""
        params = self.parameter_generator(query)
        params = params.reshape(B * Q, G, -1)
        out = x.reshape(B * Q, G, P, C)

        # 将混合参数分成两部分，一部分是M，一部分是S
        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B * Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B * Q, G, self.out_points, self.in_points)

        # M和采样特征进行混合
        """adaptive channel mixing"""
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        # S和采样特征进行混合
        """adaptive point mixing"""
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        # 将混合后的特征进行展平，然后通过全连接网络，获得最终的查询特征
        """linear transfomation to query dim"""
        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)
