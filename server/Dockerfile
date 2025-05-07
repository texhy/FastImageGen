# server/Dockerfile

# Use CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    python3.10-venv \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy and install the rest of the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/mobiusml/hqq@306e30d

# Create necessary directories with proper permissions
RUN mkdir -p /root/.cache/huggingface/hub && \
    mkdir -p /root/.cache/huggingface && \
    python3 -c "with open('/root/.cache/huggingface/token', 'w', encoding='utf-8') as f: f.write('')" && \
    chmod -R 777 /root/.cache

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    CUDA_LAUNCH_BLOCKING=1 \
    PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Set working directory
WORKDIR /app

# Copy all application files
COPY server.py worker.py image_gen_pb2.py image_gen_pb2_grpc.py image_gen.proto requirements.txt ./

# # Copy model weights directly into the container
# COPY hub/models--black-forest-labs--FLUX.1-dev /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev
# COPY hub/models--HighCWu--FLUX.1-dev-4bit /root/.cache/huggingface/hub/models--HighCWu--FLUX.1-dev-4bit

# Expose ports
EXPOSE 50051 8000

# Command to run the server
CMD ["python", "server.py"]
