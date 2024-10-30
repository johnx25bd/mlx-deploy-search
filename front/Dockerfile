FROM python:3.12-slim

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install system dependencies for building certain packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]