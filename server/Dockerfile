# server/Dockerfile

FROM python:3.10-slim

# 1) Install git (needed for your git+… deps) and clean up APT lists
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Copy requirements
COPY requirements.txt .

# 3) First install torch & friends from the PyTorch CUDA index,
#    then install all other deps from requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.7.0+cu118 torchvision torchaudio && \
    pip install --no-cache-dir -r requirements.txt

# 4) Copy the rest of your server code
COPY . .

EXPOSE 50051 8000

CMD ["python", "server.py"]
