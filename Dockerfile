FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg aria2 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user . /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && grep -v onnxruntime-gpu requirements.txt > r.txt && echo onnxruntime >> r.txt && pip install --no-cache-dir -r r.txt
USER user
ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
