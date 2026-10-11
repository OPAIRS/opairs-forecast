ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim
LABEL org.opencontainers.image.licenses="Apache-2.0" org.opencontainers.image.vendor="OPAIRS Systems FlexCo"

ARG WITH_FOUNDATION=0
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 svc

WORKDIR /srv
COPY requirements.txt requirements-foundation.txt ./
RUN pip install -r requirements.txt
# Foundation-Modelle (Chronos, zieht PyTorch). Mit --build-arg WITH_FOUNDATION=1 bauen.
# GPU-Hinweis: torch-Variante passend zu CUDA/Treiber wählen, sonst läuft Chronos auf CPU.
# TiRex-2 auf cuda braucht zusätzlich nvcc (CUDA Toolkit) im Image; in models.yaml steht es deshalb auf device: cpu.
RUN if [ "$WITH_FOUNDATION" = "1" ]; then pip install -r requirements-foundation.txt; fi

COPY app ./app
COPY models.yaml ./models.yaml

USER svc
EXPOSE 8200 8201
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fs http://localhost:8200/health || exit 1
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8200"]
