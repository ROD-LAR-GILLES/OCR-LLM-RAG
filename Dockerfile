# ──────────────────────────────
# OCR-CLI   (Python 3.11 slim)
# ──────────────────────────────
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# libs para Tesseract, pdf2image, pdfplumber y OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-spa poppler-utils ghostscript gcc \
    libglib2.0-0 libgl1-mesa-glx libsm6 libxext6 libxrender-dev \
    libgomp1 libglib2.0-0 libgtk-3-0 libavcodec-dev libavformat-dev \
    libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
    libjpeg-dev libpng-dev libtiff-dev libatlas-base-dev \
    libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev \
    libpango1.0-dev libgtk2.0-dev libgtk-3-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "interfaces/cli/main.py"]