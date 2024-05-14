import threading

# import eventlet
# import eventlet.wsgi

# eventlet.monkey_patch()

import asyncio  # noqa: E402
import concurrent.futures  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402

from quart import Quart, websocket, request, Response  # noqa: E402
from hypercorn.config import Config as hconfig  # noqa: E402
from hypercorn.asyncio import serve  # noqa: E402
from mmcv import Config  # noqa:E402

# import utils  # noqa: E402
from core import Pipeline, preprocess, Model  # noqa: E402

app = Quart(__name__)
config = hconfig().from_toml("server.toml")
memory = {}
global_index = 0
cond = threading.Condition()
executor = concurrent.futures.ProcessPoolExecutor(8)
logging.basicConfig(level=logging.INFO)
model = Model(args=config)
cfgs = Config.fromfile(config.config)
pipeline = Pipeline(args=cfgs)


@app.route("/detection", methods=["POST"])
# @utils.timer_decorator_func
async def detection():
    """Run the model on the input data and return the results.

    Inputs:
        data
    Outputs:
        results
    """
    global global_index, pipeline, model, cond, memory, executor
    data = await request.get_data()
    index, data = pickle.loads(data)
    # logging.info(f"Received {index}th data")
    if index == 0:
        memory["time"] = time.perf_counter()
        global_index = 0
    data = await preprocess(pipeline, data)
    # loop = asyncio.get_event_loop()
    # data = await loop.run_in_executor(executor, preprocess, pipeline, data)
    with cond:
        while index != global_index:
            print(index)
            cond.wait()
        if data is not None:
            results = model(data)
            global_index += 1
        cond.notify_all()
    if data is None:
        logging.info(
            f"Done sample [{global_index} / {global_index}], "
            f"fps: {0 if global_index == 0 else global_index / (time.perf_counter() - memory['time']) :.1f} sample / s"
        )
        return Response(b"", mimetype="application/octet-stream")
    data = pickle.dumps(results)
    return Response(data, mimetype="application/octet-stream")


@app.route("/test", methods=["POST"])
def test():
    def gen_result(stream):
        temp_data = bytearray()
        for chunk in stream:
            if chunk:
                temp_data.extend(chunk)
            yield temp_data
            logging.info(f"Received {len(temp_data)} bytes")
            temp_data = bytearray()

    return Response(gen_result(request.stream), mimetype="application/octet-stream")


@app.route("/info", methods=["GET"])
def info():
    return "Real-time 3D object detection server"


@app.websocket("/detection-ws")
async def detection_ws():
    """Run the model on the input data and return the results.

    Inputs:
        data
    Outputs:
        results
    """
    res_queue = asyncio.Queue()
    raw_data_queue = asyncio.Queue()
    data_queue = asyncio.Queue()

    async def sending():
        start = time.perf_counter()
        index = 0
        while True:
            result = await res_queue.get()
            if result is None:
                logging.info(
                    f"All sample Done [{index}] / ?], "
                    f"fps: {0 if index == 0 else index / (time.perf_counter() - start):.1f} sample / s"
                )
                return
            index, result = result
            if index % 20 == 0:
                logging.info(
                    f"Done sample [{index}] / ?], "
                    f"fps: {0 if index == 0 else index / (time.perf_counter() - start):.1f} sample / s"
                )
            await websocket.send(result)

    async def receiving():
        while True:
            data = await websocket.receive()
            data = pickle.loads(data)
            if len(data) == 0:
                logging.info("All data are received")
                await raw_data_queue.put(None)
                return
            await raw_data_queue.put(data)

    async def processing():
        global pre_process, executor
        while True:
            data = await raw_data_queue.get()
            if data is None:
                await data_queue.put(None)
                return
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(executor, pre_process, data)
            await data_queue.put(data)

    async def detecting():
        global model
        index = 0
        while True:
            data = await data_queue.get()
            if data is None:
                await res_queue.put(None)
                return
            index += 1
            results = model(data)
            data = pickle.dumps(results)
            await res_queue.put((index, data))

    producer = asyncio.create_task(receiving())
    processers = [asyncio.create_task(processing()) for _ in range(4)]
    consumer = asyncio.create_task(detecting())
    sender = asyncio.create_task(sending())
    await asyncio.gather(producer, *processers, consumer, sender)
    websocket.close()


async def main():
    asyncio.run(serve(app, config))


if __name__ == "__main__":
    main()
