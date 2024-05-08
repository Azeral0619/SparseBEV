import queue
import torch
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.fp16_utils import cast_tensor_type
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .utils import GridMask, pad_multiple, GpuPhotoMetricDistortion


@DETECTORS.register_module()
class SparseBEVTRT(MVXTwoStageDetector):
    def __init__(
        self,
        data_aug=None,
        stop_prev_grad=0,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        num_views=6,
    ):
        super(SparseBEVTRT, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = True
        self.num_views = num_views

        self.memory = {}
        self.queue = queue.Queue()

    @auto_fp16(apply_to=("img"), out_fp32=True)
    def extract_img_feat(self, img):
        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_feat(self, img, img_metas):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        assert img.dim() == 5

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        # move some augmentations to GPU
        if self.data_aug is not None:
            if (
                "img_color_aug" in self.data_aug
                and self.data_aug["img_color_aug"]
                and self.training
            ):
                img = self.color_aug(img)

            if "img_norm_cfg" in self.data_aug:
                img_norm_cfg = self.data_aug["img_norm_cfg"]

                norm_mean = torch.tensor(img_norm_cfg["mean"], device=img.device)
                norm_std = torch.tensor(img_norm_cfg["std"], device=img.device)

                if img_norm_cfg["to_rgb"]:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]["img_shape"] = [img_shape for _ in range(N)]
                img_metas[b]["ori_shape"] = [img_shape for _ in range(N)]

            if "img_pad_cfg" in self.data_aug:
                img_pad_cfg = self.data_aug["img_pad_cfg"]
                img = pad_multiple(
                    img, img_metas, size_divisor=img_pad_cfg["size_divisor"]
                )

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    @force_fp32(apply_to=("img", "points"))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        return self.forward_test(**kwargs)

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False):
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas[0], rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results

    def simple_test(self, img_metas, img=None, rescale=False):
        return self.simple_test_online(img_metas, img, rescale)

    def simple_test_online(self, img_metas, img=None, rescale=False):
        self.fp16_enabled = False
        assert len(img_metas) == 1  # batch_size = 1

        B, N, C, H, W = img.shape
        img = img.reshape(
            B, N // self.num_views, self.num_views, C, H, W
        )  # TODO: change num of views

        # img_filenames = img_metas[0]["filename"]
        # TODO: modify keys to timestamp
        img_timestamps = img_metas[0]["img_timestamp"]
        num_frames = len(img_timestamps) // self.num_views  # TODO: change num of views
        # assert num_frames == img.shape[1]

        img_shape = (H, W, C)
        img_metas[0]["img_shape"] = [img_shape for _ in range(len(img_timestamps))]
        img_metas[0]["ori_shape"] = [img_shape for _ in range(len(img_timestamps))]
        img_metas[0]["pad_shape"] = [img_shape for _ in range(len(img_timestamps))]

        img_feats_list, img_metas_list = [], []

        # extract feature frame by frame
        for i in range(num_frames):
            img_indices = list(
                np.arange(i * self.num_views, (i + 1) * self.num_views)
            )  # TODO: change num of views

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                if isinstance(img_metas[0][k], list):
                    img_metas_curr[0][k] = [img_metas[0][k][i] for i in img_indices]

            if img_timestamps[img_indices[0]] in self.memory:
                # found in memory
                img_feats_curr = self.memory[img_timestamps[img_indices[0]]]
            else:
                # extract feature and put into memory
                img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
                self.memory[img_timestamps[img_indices[0]]] = img_feats_curr
                self.queue.put(img_timestamps[img_indices[0]])
                while self.queue.qsize() >= 16:  # avoid OOM
                    pop_key = self.queue.get()
                    self.memory.pop(pop_key)

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)

        # reorganize
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat(
                [img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0
            )
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)

        # run detector
        bbox_list = [dict() for _ in range(1)]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox

        return bbox_list
