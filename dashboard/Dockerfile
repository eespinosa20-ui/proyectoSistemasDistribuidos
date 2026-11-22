FROM python:3.11.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY dashboard/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY dashboard/ /app/
EXPOSE 8080
HEALTHCHECK --interval=5s --timeout=3s CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/', timeout=2).close()"]
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
