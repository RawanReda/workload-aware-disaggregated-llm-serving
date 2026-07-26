import argparse
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="Disaggregated Prefill Proxy Server")
    parser.add_argument("--host", type=str, default="localhost", help="Host for the proxy server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the proxy server")
    parser.add_argument("--prefiller-port", "-p", type=int, default=8100, help="Port for the prefill server")
    parser.add_argument("--decoder-port", "-d", type=int, default=8200, help="Port for the decoder server")
    parser.add_argument("--model-name", "-m", type=str, default="<TO_DO>", help="Model name for the prefill and decoder servers")
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    prefill_url = f"http://{app.state.host}:{app.state.prefiller_port}/v1"
    decode_url = f"http://{app.state.host}:{app.state.decoder_port}/v1"

    app.state.prefill_client = httpx.AsyncClient(base_url=prefill_url)
    app.state.decode_client = httpx.AsyncClient(base_url=decode_url)

    yield

    await app.state.prefill_client.aclose()
    await app.state.decode_client.aclose()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/completions")
async def create_completions():
    # This is a placeholder for the actual implementation of the /v1/completions endpoint.
    # You would typically forward the request to the prefill and decoder servers here.
    return {"message": "This endpoint will handle completions."} #streaming response 


if __name__ == "__main__":
    args = parse_args()

    app.state.host = args.host
    app.state.prefiller_port = args.prefiller_port
    app.state.decoder_port = args.decoder_port
    app.state.model_name = args.model_name

    uvicorn.run(app, host=args.host, port=args.port)
