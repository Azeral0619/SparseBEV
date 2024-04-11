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

import utils
from models.utils import VERSION


class model(object):
    def __init__(self, args):
        # parse configs
        cfgs = Config.fromfile(args.config)

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
            utils.init_logging(None, cfgs.debug)
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

        logging.info("Creating model: %s" % cfgs.model.type)
        self.model = build_model(cfgs.model)
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

    def __call__(self, data):
        """Run the model on the input data and return the results.

        Args:
            data (dict):
                image_metas (dict | list[dict]):
                    'box_type_3d': 'LiDAR'
                    'ori_shape': [(256, 704, 3), ...]
                    'img_shape': [(256, 704, 3), ...]
                    'pad_shape': [(256, 704, 3), ...]
                    'img_timestamp': list[int]
                    'filename': list[str]
                    'lidar2img': list[np.ndarray(shape=(4, 4), dtype=float64)]
            img (torch.Tensor | list[torch.Tensor]): [3, 6, 900, 1600]

        Returns:
            result (dict):
        """
        return self.model(return_loss=False, rescale=True, **data)
