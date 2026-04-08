# Use the official Python lightweight image
FROM python:3.10-slim

# Set the working coordinate inside the container
WORKDIR /app

# Prevent Python from writing .pyc files & force unbuffered stdout logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system-level graphics dependencies (Required for OpenCV and PyTorch)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy strictly requirements first to leverage Docker's aggressive caching
COPY requirements.txt .

# Install core dependencies. We use pip with --no-cache-dir to keep the image pure/small
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application codebase
COPY . .

# Expose the precise port that JalDrishti runs its Flask API on
EXPOSE 5500

# Fire the primary API script
CMD ["python", "api.py"]
