FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gmeet_pipeline/ ./gmeet_pipeline/

EXPOSE 9120

CMD ["python", "-m", "gmeet_pipeline.main"]
