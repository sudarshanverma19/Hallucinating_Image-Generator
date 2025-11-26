# Multi-stage build for optimized Enhanced Deep Dreams
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results temp

# Set environment variables for enhanced performance
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Optimize TensorFlow for CPU
ENV TF_NUM_INTRAOP_THREADS=0
ENV TF_NUM_INTEROP_THREADS=0

# Remove unnecessary files to reduce image size
RUN find . -name "*.pyc" -delete && \
    find . -name "__pycache__" -delete && \
    find . -name "*.pyo" -delete

# Expose the port
EXPOSE 8000

# Enhanced health check
HEALTHCHECK --interval=30s --timeout=45s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the enhanced application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop"]