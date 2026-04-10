FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    pydantic>=2.0.0 \
    openai>=1.0.0 \
    httpx>=0.24.0

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
