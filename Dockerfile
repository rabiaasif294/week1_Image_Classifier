
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

COPY requirements-docker.txt .

RUN python -m pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-docker.txt

COPY src/ ./src/

EXPOSE 8502

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8502"]
