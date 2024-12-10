FROM python:3.11-slim

WORKDIR /app

COPY finetune/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download nltk data
RUN python -m nltk.downloader stopwords

RUN mkdir -p /app/data

# Copy the application code
COPY finetune /app/
COPY utils /app/utils

# Run the finetuning script
CMD ["python", "main.py"]