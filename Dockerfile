FROM python:3.10

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# ===============================
# System dependencies (ML safe)
# ===============================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ===============================
# Python tooling
# ===============================
RUN pip install --upgrade pip setuptools wheel

# ===============================
# Install dependencies (ORDERED)
# ===============================
COPY requirements-base.txt .
RUN pip install -r requirements-base.txt

COPY requirements-ml.txt .
RUN pip install --prefer-binary -r requirements-ml.txt


# ===============================
# Copy application code
# ===============================
COPY . .

EXPOSE 8000

CMD ["python", "src/predict.py"]
