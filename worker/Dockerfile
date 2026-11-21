FROM python:3.11.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY worker/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY scripts/prepare_data.py /app/prepare_data.py
COPY worker/ /app/
HEALTHCHECK --interval=5s CMD ["python", "health.py"]
CMD ["python", "main.py"]
