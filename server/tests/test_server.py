import os
import sys
import time
import pytest
import grpc
import multiprocessing as mp
import threading
# Add the server directory to PYTHONPATH so we can import the generated pb files and server
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import image_gen_pb2
import image_gen_pb2_grpc
from server import serve, task_queue, result_queue

SERVER_ADDRESS = "localhost:50051"
API_KEY        = "client1"
INVALID_KEY    = "badkey"

@pytest.fixture(scope="module", autouse=True)
def grpc_server():
    """Starts the gRPC server in a subprocess for tests."""
    proc = mp.Process(target=serve)
    proc.start()
    time.sleep(2)  # give server time to start
    yield
    proc.terminate()
    proc.join()

def make_stub(api_key=API_KEY):
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = image_gen_pb2_grpc.ImageGenStub(channel)
    metadata = [("api-key", api_key)]
    return stub, metadata

def test_ping_success():
    stub, meta = make_stub()
    resp = stub.Ping(image_gen_pb2.PingRequest(), metadata=meta)
    assert resp.message == "Pong"

def test_ping_invalid_api_key():
    stub, meta = make_stub(INVALID_KEY)
    with pytest.raises(grpc.RpcError) as exc:
        stub.Ping(image_gen_pb2.PingRequest(), metadata=meta)
    assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED

@pytest.mark.parametrize("h,w,s,g,err", [
    (0,512,5,1.0,"height"),  
    (512,0,5,1.0,"width"),
    (512,512,0,1.0,"num_inference_steps"),
    (512,512,21,1.0,"num_inference_steps"),
    (512,512,5,0.0,"guidance_scale"),
    (512,512,5,11.0,"guidance_scale"),
])
def test_generate_param_validation(h, w, s, g, err):
    stub, meta = make_stub()
    req = image_gen_pb2.GenerateRequest(
        prompt="test", height=h, width=w,
        num_inference_steps=s, guidance_scale=g
    )
    with pytest.raises(grpc.RpcError) as exc:
        stub.Generate(req, metadata=meta)
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert err in exc.value.details()

def test_generate_invalid_api_key():
    stub, meta = make_stub(INVALID_KEY)
    req = image_gen_pb2.GenerateRequest(
        prompt="test", height=128, width=128,
        num_inference_steps=5, guidance_scale=1.0
    )
    with pytest.raises(grpc.RpcError) as exc:
        stub.Generate(req, metadata=meta)
    assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED

def test_generate_success_png_header():
    stub, meta = make_stub()
    req = image_gen_pb2.GenerateRequest(
        prompt="A cat", height=64, width=64,
        num_inference_steps=1, guidance_scale=1.0
    )
    resp = stub.Generate(req, metadata=meta)
    # PNG files start with \x89PNG\r\n\x1a\n
    assert resp.image_png.startswith(b"\x89PNG\r\n\x1a\n")

def test_single_user_queue_exhausted():
    stub, meta = make_stub()

    # 1) Start a background thread that sends a long prompt
    done = threading.Event()
    def long_call():
        req_long = image_gen_pb2.GenerateRequest(
            prompt="this prompt will hang", height=64, width=64,
            num_inference_steps=2, guidance_scale=1.0
        )
        # This will acquire the GPU lock and block the server until done
        _ = stub.Generate(req_long, metadata=meta)
        done.set()

    t = threading.Thread(target=long_call, daemon=True)
    t.start()

    # 2) Give it a moment to acquire the lock
    time.sleep(0.1)

    # 3) Now the lock is held—second call should immediately fail
    req2 = image_gen_pb2.GenerateRequest(
        prompt="x", height=64, width=64,
        num_inference_steps=1, guidance_scale=1.0
    )
    with pytest.raises(grpc.RpcError) as exc:
        stub.Generate(req2, metadata=meta)
    assert exc.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED

    # 4) Let the background thread finish so it doesn’t leak
    done.set()
    t.join(timeout=5)

def test_multiple_clients_queue_positions():
    stub1, meta1 = make_stub()
    stub2, meta2 = make_stub()

    # 1) First client grabs the lock
    done = threading.Event()
    def first_call():
        req1 = image_gen_pb2.GenerateRequest(
            prompt="first", height=64, width=64,
            num_inference_steps=2, guidance_scale=1.0
        )
        _ = stub1.Generate(req1, metadata=meta1)
        done.set()

    t1 = threading.Thread(target=first_call, daemon=True)
    t1.start()
    time.sleep(0.1)

    # 2) Second client immediately fails
    req2 = image_gen_pb2.GenerateRequest(
        prompt="second", height=64, width=64,
        num_inference_steps=1, guidance_scale=1.0
    )
    with pytest.raises(grpc.RpcError) as exc:
        stub2.Generate(req2, metadata=meta2)
    assert exc.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED

    t1.join(timeout=5)