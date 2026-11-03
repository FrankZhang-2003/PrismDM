FROM python:3.10-slim-bookworm
WORKDIR /workspace/PrismDM
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-mpc.txt ./
RUN pip install --no-cache-dir -r requirements-mpc.txt
COPY . .
ENV JAX_PLATFORMS=cpu
CMD ["bash"]
