import argparse
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="Disaggregated Prefill Proxy Server")
    parser.add_argument("--host", type=str, default="localhost", help="Host for the proxy server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the proxy server")
    parser.add_argument("--prefiller-port", "-p", type=int, default=8100, help="Port for the prefill server")
    parser.add_argument("--decoder-port", "-d", type=int, default=8200, help="Port for the decoder server")
    parser.add_argument("--model-name", "-m", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name for the prefill and decoder servers")
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    prefill_url = f"http://{app.state.host}:{app.state.prefiller_port}/v1"
    decode_url = f"http://{app.state.host}:{app.state.decoder_port}/v1"

    app.state.prefill_client = httpx.AsyncClient(
        timeout=None,
        base_url=prefill_url,
        limits=httpx.Limits(
            max_connections=None,
            max_keepalive_connections=None,
        ),
    )
    app.state.decode_client = httpx.AsyncClient(
        timeout=None,
        base_url=decode_url,
        limits=httpx.Limits(
            max_connections=None,
            max_keepalive_connections=None,
        ),
    )

    yield

    await app.state.prefill_client.aclose()
    await app.state.decode_client.aclose()


app = FastAPI(lifespan=lifespan)



@app.get("/v1/models")
async def get_models():
    try:
        prefill_models_response = await app.state.prefill_client.get("/models")
        prefill_models_response.raise_for_status()
        return prefill_models_response.json()
    except Exception as e:
        return {"error": f"Failed to fetch models from prefill service: {str(e)}"}
   

@app.post("/v1/completions")
async def create_completions(request: Request):

    try:
        req_data = await request.json()

        prefill_prepare_request = req_data.copy()

        # Set max_tokens to 1 is a signal to the prefill service that we only want the prefill step and and the kv cache.
        # The decoder service will handle the rest of the token generation.
        prefill_prepare_request["max_tokens"] = 1

        prefill_response = await app.state.prefill_client.post("/completions", json=prefill_prepare_request)
        prefill_response.raise_for_status()
 

        async def async_stream_generator():
            async with app.state.decode_client.stream("POST", "/completions", json=req_data) as stream_response:
                stream_response.raise_for_status()
                async for chunk in stream_response.aiter_bytes():
                    yield chunk

        return StreamingResponse(
            async_stream_generator(),
            media_type="text/event-stream",
        )

    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    args = parse_args()

    app.state.host = args.host
    app.state.prefiller_port = args.prefiller_port
    app.state.decoder_port = args.decoder_port
    app.state.model_name = args.model_name

    uvicorn.run(app, host=args.host, port=args.port)
