import argparse
import pickle
import time
import zlib
from gevent import pywsgi

from flask import Flask, Response, request
import logging

import torch

from core import model

app = Flask(__name__)
core = None


@app.route("/detection", methods=["POST"])
def run_model():
    """Run the model on the input data and return the results.

    Inputs:
        data
    Outputs:
        results
    """

    def gen_result(stream):
        global core
        temp_data = bytearray()
        i = 0
        start = time.perf_counter()
        while True:
            chunk = stream.read(4 * 1024 * 1024)
            if not chunk:
                break
            temp_data.extend(chunk)
            while b"--data-boundary--" in temp_data:
                if i != 0:
                    yield b"--result-boundary--"
                obj, temp_data = temp_data.split(b"--data-boundary--", 1)
                obj = pickle.loads(zlib.decompress(obj))

                with torch.no_grad():
                    torch.cuda.synchronize()
                    logging.info(f"Processing {i}th data")
                    results = core(obj)
                    torch.cuda.synchronize()
                if i != 0 and i % 20 == 0:
                    logging.info(
                        f"Done sample [{i} / ?], "
                        f"fps: {(time.perf_counter() - start) / i:.1f} sample / s"
                    )
                obj = pickle.dumps(results)
                yield obj
                i += 1
        logging.info(
            f"Done sample [{i} / {i}], "
            f"fps: {(time.perf_counter() - start) / i:.1f} sample / s"
        )

    return Response(gen_result(request.stream), mimetype="application/octet-stream")


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
    server = pywsgi.WSGIServer(("0.0.0.0", args.port), app)
    server.serve_forever()


if __name__ == "__main__":
    main()
