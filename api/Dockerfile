
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Download nltk data
RUN python -m nltk.downloader stopwords

# Copy the data files
COPY data/word-vector-embeddings.model /app/data/word-vector-embeddings.model
COPY data/two_tower_state_dict.pth /app/data/two_tower_state_dict.pth
COPY data/doc-index-64.faiss /app/data/doc-index-64.faiss
COPY data/training-with-tokens.parquet /app/data/training-with-tokens.parquet

# Copy the current directory contents into the container at /app
COPY . /app/

# Expose the port that the application listens on
EXPOSE 8000

# Command to run the Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
