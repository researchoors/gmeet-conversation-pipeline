FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY meeting_agent.py .

EXPOSE 9120

CMD ["python", "meeting_agent.py"]
