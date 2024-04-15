import argparse
import logging
import pickle
import time
import zlib

import torch
from flask import Flask, Response, request
from flask_socketio import SocketIO, emit, disconnect

from core import model

app = Flask(__name__)
socketio = SocketIO(logger=True, ping_timeout=600)
socketio.init_app(app, cors_allowed_origins="*")
core = None
memory = {}


@app.route("/detection", methods=["POST"])
def detection():
    """Run the model on the input data and return the results.

    Inputs:
        data
    Outputs:
        results
    """
    data = request.data
    data = pickle.loads(zlib.decompress(data))
    with torch.no_grad():
        torch.cuda.synchronize()
        logging.info("Processing data")
        results = core(data)
        torch.cuda.synchronize()
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


@socketio.on("detection")
def detection_ws(data):
    """Run the model on the input data and return the results.

    Inputs:
        data
    Outputs:
        results
    """
    global memory
    if len(data) == 0:
        logging.info("All data are received")
        disconnect()
        return
    data = pickle.loads(zlib.decompress(data))
    memory[request.sid]["count"] += 1
    with torch.no_grad():
        torch.cuda.synchronize()
        logging.info(f"Processing {memory[request.sid]['count']}th data")
        results = core(data)
        torch.cuda.synchronize()
    if memory[request.sid]["count"] != 0 and memory[request.sid]["count"] % 20 == 0:
        logging.info(
            f"Done sample [{memory[request.sid]['count']} / ?], "
            f"fps: {0 if memory[request.sid]['count'] == 0 else (time.perf_counter() - memory[request.sid]['time']) / memory[request.sid]['count']:.1f} sample / s"
        )
    data = pickle.dumps(results)
    emit("result", data)


@socketio.on("connect")
def handle_connect():
    global memory
    memory[request.sid] = {}
    memory[request.sid]["time"] = time.perf_counter()
    memory[request.sid]["count"] = 0
    logging.info("Connected to client")
    # emit("data")


@socketio.on("pong")
def pong():
    emit("ping")


@socketio.on("disconnect")
def handle_disconnect():
    global memory
    logging.info(
        f"Done sample [{memory[request.sid]['count']} / {memory[request.sid]['count']}], "
        f"fps: {0 if memory[request.sid]['count'] == 0 else (time.perf_counter() - memory[request.sid]['time']) / memory[request.sid]['count']:.1f} sample / s"
    )
    del memory[request.sid]
    logging.info("Disconnected from client")
    disconnect()


def main():
    global core
    parser = argparse.ArgumentParser(description="Validate a detector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    core = model(args=args)
    socketio.run(app, host="0.0.0.0", port=args.port, debug=True)
    # server = pywsgi.WSGIServer(("0.0.0.0", args.port), app)
    # server.serve_forever()


if __name__ == "__main__":
    main()
