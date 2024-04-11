import argparse
import asyncio
import importlib
import logging
import pickle
from queue import Queue
import signal
import tkinter as tk
import httpx
import socketio
import zlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataloader, build_dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import box_in_image
from PIL import Image, ImageTk
from pyquaternion import Quaternion
from sympy import true

from viz_bbox_predictions import convert_to_nusc_box

classname_to_color = {  # RGB
    "car": (255, 158, 0),  # Orange
    "pedestrian": (0, 0, 230),  # Blue
    "trailer": (255, 140, 0),  # Darkorange
    "truck": (255, 99, 71),  # Tomato
    "bus": (255, 127, 80),  # Coral
    "motorcycle": (255, 61, 99),  # Red
    "construction_vehicle": (233, 150, 70),  # Darksalmon
    "bicycle": (220, 20, 60),  # Crimson
    "barrier": (112, 128, 144),  # Slategrey
    "traffic_cone": (47, 79, 79),  # Darkslategrey
}

root = tk.Tk()
window_size = (1920, 1080)
root.geometry(f"{window_size[0]}x{window_size[1]}")
label = tk.Label(root)
label.pack()
parser = argparse.ArgumentParser()
val_dataset = None
val_loader = None
nusc = None
pool_render = ThreadPoolExecutor(1)
pool_request = ThreadPoolExecutor(1)
args = None
queue = Queue()
resp_queue = Queue()
interrupted = False
sio = socketio.Client()
client = httpx.Client()
total = None
responses = []


def update_image(image):
    global label, window_size
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = image.resize(window_size)
    image = ImageTk.PhotoImage(image)
    label.configure(image=image)
    label.image = image


def viz_bbox_cv(nusc, bboxes, data_info):
    cam_types = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
    ]

    canvas = None
    margin = 20

    for cam_id, cam_type in enumerate(cam_types):
        sample_data_token = nusc.get("sample", data_info["token"])["data"][cam_type]

        sd_record = nusc.get("sample_data", sample_data_token)
        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        intrinsic = np.array(cs_record["camera_intrinsic"])

        img_path = nusc.get_sample_data_path(sample_data_token)
        img_size = (sd_record["width"], sd_record["height"])

        img = cv2.imread(img_path)
        width, height = img.shape[1], img.shape[0]

        if canvas is None:
            canvas = np.zeros(
                (height * 2 + margin, width * 3 + margin * 2, 3), dtype=np.uint8
            )

        for bbox in bboxes:
            bbox = bbox.copy()

            # Move box to ego vehicle coord system
            bbox.rotate(Quaternion(data_info["lidar2ego_rotation"]))
            bbox.translate(np.array(data_info["lidar2ego_translation"]))

            # Move box to sensor coord system
            bbox.translate(-np.array(cs_record["translation"]))
            bbox.rotate(Quaternion(cs_record["rotation"]).inverse)

            if box_in_image(bbox, intrinsic, img_size):
                c = classname_to_color[bbox.name]
                bbox.render_cv2(
                    img, view=intrinsic, normalize=True, colors=(c, c, c), linewidth=2
                )

        x, y = cam_id % 3 * (width + margin), cam_id // 3 * (height + margin)
        canvas[y : y + height, x : x + width] = img

    update_image(canvas)


def handle_request(url, data):
    global client
    response = client.post(
        url, content=data, headers={"Content-Type": "application/octet-stream"}
    )
    return pickle.loads(response.content)


def generate_stream_data():
    global val_dataset, val_loader
    global nusc
    # parse configs
    cfgs = Config.fromfile(args.config)

    # register custom module
    importlib.import_module("loaders")

    set_random_seed(0, deterministic=True)

    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    if "mini" in cfgs.data.val.ann_file:
        nusc = NuScenes(
            version="v1.0-mini", dataroot=cfgs.data.val.data_root, verbose=False
        )
    else:
        nusc = NuScenes(
            version="v1.0-trainval", dataroot=cfgs.data.val.data_root, verbose=False
        )
    logging.info("Data loaded")

    for i, data in enumerate(val_loader):
        logging.info(f"Sending {i}th data")
        resp_queue.put(handle_request(args.url, zlib.compress(pickle.dumps(data))))
        queue.put(i)


def generate_test_stream_data():
    global interrupted

    def signal_handler(sig, frame):
        global interrupted
        interrupted = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, signal_handler)

    while not interrupted:
        # 无限循环 a-z
        for i in range(97, 123):
            yield bytes([i])


def render_response(result):
    results = result[0]["pts_bbox"]
    bboxes_pred = convert_to_nusc_box(
        bboxes=results["boxes_3d"].tensor.numpy(),
        scores=results["scores_3d"].numpy(),
        labels=results["labels_3d"].numpy(),
        score_threshold=args.score_threshold,
        lift_center=True,
    )
    i = queue.get()
    viz_bbox_cv(nusc, bboxes_pred, val_dataset.data_infos[i])


def handle_response():
    global val_loader
    for _ in val_loader:
        render_response(resp_queue.get())


@sio.event
def connect():
    logging.info("Connected to server")


@sio.event
def disconnect():
    global total, queue
    total = None
    logging.info("Disconnected from server")
    queue = Queue()


@sio.on("data")
def generate_data_ws():
    global val_dataset, nusc, sio, total
    # parse configs
    cfgs = Config.fromfile(args.config)

    # register custom module
    importlib.import_module("loaders")

    set_random_seed(0, deterministic=True)

    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    if "mini" in cfgs.data.val.ann_file:
        nusc = NuScenes(
            version="v1.0-mini", dataroot=cfgs.data.val.data_root, verbose=False
        )
    else:
        nusc = NuScenes(
            version="v1.0-trainval", dataroot=cfgs.data.val.data_root, verbose=False
        )
    logging.info("Data loaded")

    for i, data in enumerate(val_loader):
        queue.put(i)
        logging.info(f"Sending {i}th data")
        sio.emit("detection", zlib.compress(pickle.dumps(data)))

    total = len(val_loader)


@sio.event
def result(data):
    result, count = pickle.loads(data)
    # logging.info(f"Rendering {count}th data")
    render_response(result)
    if count == total:
        sio.disconnect()


def main():
    global parser, args, sio, root, pool_request
    logging.getLogger().setLevel(logging.INFO)
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8080/detection")
    parser.add_argument("--config", required=true)
    parser.add_argument("--score_threshold", default=0.3)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--enable_ws", type=int, default=0)
    args = parser.parse_args()

    if args.enable_ws == 1:
        url = args.url.replace("/detection", "")  # .replace("http", "ws")
        sio.connect(url, transports=["websocket"])
        # pool_request.submit(generate_data_ws())
    else:
        if args.test == 0:
            pool_request.submit(generate_stream_data)
            pool_render.submit(handle_response)
        else:

            def task():
                url = args.url.replace("detection", "test")
                response = requests.post(
                    url, data=generate_test_stream_data(), stream=True
                )
                for chunk in response.iter_content(chunk_size=512):
                    if chunk:
                        print(chunk, flush=True)

            pool_request.submit(task()).result()
    root.mainloop()


if __name__ == "__main__":
    main()
