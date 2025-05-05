# ⚡ FLUX Image Generation Server

A high-performance image generation server powered by [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), served over gRPC and monitored with HTTP metrics. Built for GPU-accelerated environments using Docker, PyTorch, and NVIDIA CUDA.

## 🚀 Features

- **gRPC API** for image generation using FLUX.1-dev
- **Asynchronous multiprocessing** for scalable worker management
- **Automatic quantized model loading**
- **Metrics endpoint** for real-time CPU, RAM, GPU, and worker status
- **CUDA-enabled** with torch, torchvision, torchaudio pre-installed
- **Model caching and pre-loading** using Hugging Face Hub
- **Secure API key-based access control**

---

## 🐳 Docker Setup

Ensure you have **NVIDIA Docker runtime** enabled for GPU support.

### 1. Build the Docker image

```bash
docker build -t flux-server -f server/Dockerfile .
docker run --gpus all -p 50051:50051 -p 8000:8000 \
    -e API_KEYS="client1,client2" \
    flux-server
```
API Overview
gRPC Methods
Ping: Health check, returns Pong

Generate: Generates an image based on a prompt and parameters
| Parameter             | Type   | Range      |
| --------------------- | ------ | ---------- |
| `prompt`              | string | any prompt |
| `height`              | int    | 1–1024     |
| `width`               | int    | 1–1024     |
| `num_inference_steps` | int    | 1–20       |
| `guidance_scale`      | float  | 0.1–10.0   |


api-key: client1

Sample gRPC Client (Python)
```bash
import grpc
import image_gen_pb2, image_gen_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = image_gen_pb2_grpc.ImageGenStub(channel)

metadata = [("api-key", "client1")]
response = stub.Generate(
    image_gen_pb2.GenerateRequest(
        prompt="a futuristic cityscape",
        height=512,
        width=512,
        num_inference_steps=15,
        guidance_scale=7.5
    ), metadata=metadata
)

with open("output.png", "wb") as f:
    f.write(response.image_png)
```
``` bash
├── server/
│   ├── Dockerfile
├── server.py              # gRPC + HTTP server
├── worker.py              # Model inference worker process
├── image_gen.proto        # gRPC definitions
├── image_gen_pb2.py       # Generated gRPC code
├── image_gen_pb2_grpc.py  # Generated gRPC server/client
├── requirements.txt       # Python dependencies
├── hub/                   # Pre-downloaded Hugging Face models
```

``` bash
pip install -r requirements.txt
pip install git+https://github.com/mobiusml/hqq@306e30d
```

Test Cases (Implemented in server/tests)
🧪 Test Suite Overview
This project includes a comprehensive set of tests for the gRPC image generation server to ensure correct functionality, security, and robustness. The tests use pytest and are located in the test file associated with server.py.

Server Startup
The gRPC server is launched as a subprocess using a pytest fixture before the test suite runs:
```bash
@pytest.fixture(scope="module", autouse=True)
def grpc_server():
    ...
```
Test Cases
| Test Name                               | Purpose                                                                   | Expected Behavior                                                      |
| --------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `test_ping_success`                     | Verify successful ping with valid API key.                                | Returns `"Pong"` string.                                               |
| `test_ping_invalid_api_key`             | Ensure server rejects unauthorized ping requests.                         | Raises `UNAUTHENTICATED` gRPC error.                                   |
| `test_generate_param_validation`        | Check that invalid parameters (height, width, steps, scale) are rejected. | Raises `INVALID_ARGUMENT` with specific error message.                 |
| `test_generate_invalid_api_key`         | Ensure image generation requires a valid API key.                         | Raises `UNAUTHENTICATED` error.                                        |
| `test_generate_success_png_header`      | Validate successful image generation and output format.                   | Response image starts with valid PNG header bytes.                     |
| `test_single_user_queue_exhausted`      | Enforce one active request per client to prevent concurrent GPU use.      | Second request during active job raises `RESOURCE_EXHAUSTED`.          |
| `test_multiple_clients_queue_positions` | Ensure server locks GPU across clients to enforce exclusivity.            | Second client request fails with `RESOURCE_EXHAUSTED` during conflict. |


API Authentication
Each gRPC request must include a valid "api-key" metadata field. Invalid keys are rejected with UNAUTHENTICATED errors

Concurrency Handling
Tests simulate realistic multithreading scenarios to validate GPU/resource locking:

Requests from the same or different clients are serialized.

Concurrent access is gracefully blocked and reported with RESOURCE_EXHAUSTED.

Running the Tests
To execute the test suite:
```bash
pytest test_server.py
```


Contributing
Pull requests, issues, and suggestions are welcome! If you're using this in production or research, feel free to open an issue and share your use case.

Acknowledgements
Black Forest Labs
HighCWu for quantized FLUX model
MobiusML for HQQ library



