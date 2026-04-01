FROM python:3.10-slim

WORKDIR /app
COPY . /app

EXPOSE 7860

CMD ["python", "app.py"]