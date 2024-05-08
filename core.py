import importlib
import logging
import os

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.models import build_model
import threading
import utils
from models.utils import VERSION
from loaders.pipelines import (
    CustomLoadMultiViewImageFromFiles,
    LoadMultiViewImageFromMultiSweeps,
    RandomTransformImage,
)
from mmdet3d.datasets.pipelines import (
    MultiScaleFlipAug3D,
    DefaultFormatBundle3D,
    Collect3D,
)


class model(object):
    def __init__(self, args):
        self.mutex = threading.Lock()
        # parse configs
        self.cfgs = Config.fromfile(args.config)

        # register custom module
        importlib.import_module("models")
        importlib.import_module("loaders")

        # MMCV, please shut up
        from mmcv.utils.logging import logger_initialized

        logger_initialized["root"] = logging.Logger(__name__, logging.WARNING)
        logger_initialized["mmcv"] = logging.Logger(__name__, logging.WARNING)

        # you need GPUs
        assert torch.cuda.is_available()

        # determine local_rank and world_size
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(args.local_rank)

        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(args.world_size)

        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        if local_rank == 0:
            utils.init_logging(None, self.cfgs.debug)
        else:
            logging.root.disabled = True

        logging.info("Using GPU: %s" % torch.cuda.get_device_name(local_rank))
        torch.cuda.set_device(local_rank)

        if world_size > 1:
            logging.info("Initializing DDP with %d GPUs..." % world_size)
            dist.init_process_group("nccl", init_method="env://")

        logging.info("Setting random seed: 0")
        set_random_seed(0, deterministic=True)
        cudnn.benchmark = True

        logging.info("Creating model: %s" % self.cfgs.model.type)
        self.model = build_model(self.cfgs.model)
        self.model.cuda()
        self.model.fp16_enabled = True

        if world_size > 1:
            self.model = MMDistributedDataParallel(
                self.model, [local_rank], broadcast_buffers=False
            )
        else:
            self.model = MMDataParallel(self.model, [0])

        logging.info("Loading checkpoint from %s" % args.weights)
        checkpoint = load_checkpoint(
            self.model,
            args.weights,
            map_location="cuda",
            strict=True,
            logger=logging.Logger(__name__, logging.ERROR),
        )
        if "version" in checkpoint:
            VERSION.name = checkpoint["version"]
        self.model.eval()

    @utils.timer_decorator
    def __call__(self, data):
        """Run the model on the input data and return the results.

        Args:
            data (dict):
                image_metas (dict | list[dict]):
                    'img_timestamp': list[int]
                    'lidar2img': list[np.ndarray(shape=(4, 4), dtype=float64)]
                img (torch.Tensor | list[torch.Tensor]): [3, 6, 900, 1600]

        Returns:
            result (dict):
        """
        self.mutex.acquire()
        try:
            with torch.no_grad():
                torch.cuda.synchronize()
                res = self.model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()
        except Exception as e:
            self.mutex.release()
            raise e
        self.mutex.release()
        return res


class PreProcess(object):
    def __init__(self, args):
        self.cfgs = Config.fromfile(args.config)
        num_frames = self.cfgs.num_frames
        num_views = self.cfgs.num_views
        ida_aug_conf = self.cfgs.ida_aug_conf
        class_names = self.cfgs.class_names
        self.customLoadMultiViewImageFromFiles = CustomLoadMultiViewImageFromFiles(
            to_float32=False,
            color_type="color",
            num_views=num_views,
            from_base64=True,
        )
        self.loadMultiViewImageFromMultiSweeps = LoadMultiViewImageFromMultiSweeps(
            sweeps_num=num_frames - 1,
            test_mode=True,
            num_views=num_views,
        )
        self.randomTransformImage = RandomTransformImage(
            ida_aug_conf=ida_aug_conf,
            training=False,
            num_views=num_views,
        )
        self.multiScaleFlipAug3D = MultiScaleFlipAug3D(
            img_scale=(1600, 900),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                DefaultFormatBundle3D(class_names=class_names, with_label=False),
                Collect3D(
                    keys=["img"],
                    meta_keys=(
                        "lidar2img",
                        "img_timestamp",
                    ),
                ),
            ],
        )

    @utils.timer_decorator
    def __call__(self, data):
        """Preprocess the input data.

        Args:
            data (dict):
                - img (list[str]): Multi-view image(base64) arrays.
                - img_timestamp (list[float]): Timestamps of images.
                - lidar2img (list[np.ndarray]): Lidar to image transformation matrices.

        Returns:
            _type_: _description_
        """
        if data is None:
            return data

        # 假设 data 是你想打印类型的数据
        data = self.customLoadMultiViewImageFromFiles(data)
        data = self.loadMultiViewImageFromMultiSweeps(data)
        data = self.randomTransformImage(data)
        data = self.multiScaleFlipAug3D(data)
        data["img"][0]._data = [
            data["img"][0]._data.reshape([1] + list(data["img"][0]._data.shape))
        ]
        data["img_metas"][0]._data = [[data["img_metas"][0]._data]]
        return data
