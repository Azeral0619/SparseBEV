import argparse
import importlib
import logging
import pickle
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import cv2
import httpx
import numpy as np
import socketio
import websocket
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import box_in_image
from PIL import Image, ImageTk
from pyquaternion import Quaternion

from loaders.pipelines.loading import LoadImageToBase64
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
pool_request = ThreadPoolExecutor(8)
args = None
num_views = None
queue = Queue()
resp_queue = Queue()
sio = socketio.Client(logger=True)
client = httpx.Client()


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


def handle_request(url, index, data, end=False):
    global client
    if end:
        logging.info("All data are sent!")
    else:
        logging.info(f"Sending {index}th data")
    response = client.post(
        url,
        content=data,
        headers={"Content-Type": "application/octet-stream"},
    )
    if end:
        return
    result = pickle.loads(response.content)
    queue.put(index)
    render_response(result)


def generate_stream_data():
    global val_dataset, client
    index = 0
    loadImageToBase64 = LoadImageToBase64(num_views=6)
    for i in range(len(val_dataset)):
        data = val_dataset.get_data_info(i)
        data = loadImageToBase64(data)
        index = i
        data = pickle.dumps((i, data))
        pool_render.submit(handle_request, args.url, i, data)
    index += 1
    data = pickle.dumps((index, None))
    pool_render.submit(handle_request, args.url, i, data, True)


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


@sio.on("connect")
def connect():
    logging.info("Connected to server")


@sio.on("disconnect")
def disconnect():
    global queue
    logging.info("Disconnected from server")
    queue = Queue()


@sio.on("ping")
def ping():
    sio.emit("pong")


def load_data():
    global val_dataset, val_loader, nusc, sio, num_views
    # parse configs
    cfgs = Config.fromfile(args.config)
    num_views = cfgs.num_views

    # register custom module
    importlib.import_module("loaders")

    set_random_seed(0, deterministic=True)

    val_dataset = build_dataset(cfgs.data.val)

    if "mini" in cfgs.data.val.ann_file:
        nusc = NuScenes(
            version="v1.0-mini", dataroot=cfgs.data.val.data_root, verbose=False
        )
    else:
        nusc = NuScenes(
            version="v1.0-trainval", dataroot=cfgs.data.val.data_root, verbose=False
        )
    logging.info("Data loaded")


@sio.on("data")
def generate_data_ws():
    global val_dataset, queue, sio, num_views
    loadImageToBase64 = LoadImageToBase64(num_views=num_views)
    for i in range(len(val_dataset)):
        data = val_dataset.get_data_info(i)
        data = loadImageToBase64(data)
        queue.put(i)
        logging.info(f"Sending {i}th data")
        if not sio.connected:
            return
        sio.emit("detection", pickle.dumps(data))

    sio.emit("detection", "")


@sio.on("result")
def result(data):
    result = pickle.loads(data)
    # logging.info(f"Rendering {count}th data")
    render_response(result)


def main():
    global parser, args, sio, root, pool_request
    logging.basicConfig(level=logging.INFO)
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8080/detection")
    parser.add_argument("--config", required=True)
    parser.add_argument("--score_threshold", default=0.3)
    parser.add_argument("--enable_ws", type=int, default=0)
    args = parser.parse_args()
    load_data()

    if args.enable_ws == 1:
        url = args.url.replace("/detection", "")  # .replace("http", "ws")
        sio.connect(url)
        # pool_request.submit(generate_data_ws)
    else:
        pool_request.submit(generate_stream_data)

    root.mainloop()


if __name__ == "__main__":
    main()
