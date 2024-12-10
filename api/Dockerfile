
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

COPY api/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Download nltk data
RUN python -m nltk.downloader stopwords

RUN mkdir -p /app/data

# Copy the current directory contents into the container at /app
COPY api /app/
COPY utils /app/utils

# Expose the port that the application listens on
EXPOSE 8000

# Command to run the Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
