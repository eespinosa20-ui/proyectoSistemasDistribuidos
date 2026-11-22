FROM python:3.11.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY coordinator/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY coordinator/ /app/
EXPOSE 50051
HEALTHCHECK --interval=5s --timeout=3s --retries=3 --start-period=10s CMD ["python", "main.py", "health"]
CMD ["python", "main.py"]
