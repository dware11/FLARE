FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/backend \
    XDG_CACHE_HOME=/tmp/.cache \
    TORCH_HOME=/tmp/.cache/torch \
    MPLCONFIGDIR=/tmp/.cache/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt \
    && pip install gunicorn

COPY . /app

RUN useradd -m -u 10001 flare \
    && mkdir -p /app/backend/static/uploads /app/backend/static/masks /app/backend/static/cam \
    && chown -R flare:flare /app/backend/static \
    && mkdir -p /home/flare/.cache \
    && chown -R flare:flare /home/flare

USER flare

EXPOSE 5000

CMD ["gunicorn", \
     "--workers", "1", \
     "--threads", "8", \
     "--timeout", "180", \
     "--worker-tmp-dir", "/tmp", \
     "--bind", "0.0.0.0:5000", \
     "backend.app:app"]
