# Quick FastAPI Boilerplate

This is a basic FastAPI project template, designed to quickly start building APIs with Python. The project is structured with modularity and scalability in mind.

## Project Structure

```plaintext
fastapi_app/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   └── models/
│       └── __init__.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.7+
- `pip` (Python package manager)
- (Optional) Virtual environment setup tool (`venv` or `virtualenv`)

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/johnx25bd/fastapi-boilerplate.git
   cd fastapi-boilerplate
   ```

2. **Create and Activate a Virtual Environment**

   It’s recommended to use a virtual environment to keep dependencies isolated.

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   Install all necessary packages using the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables (Optional)**

   If your application requires environment variables, set them up in a `.env` file in the root directory.

   Example `.env`:

   ```plaintext
   DATABASE_URL=postgresql://user:password@localhost/dbname
   SECRET_KEY=your_secret_key
   ```

## Running the Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --reload
```

- `app.main:app` specifies the `app` instance in `main.py`.
- `--reload` enables hot-reloading, useful for development.

After starting the server, you should see the output indicating that it’s running at `http://127.0.0.1:8000`.

## API Documentation

FastAPI automatically generates interactive documentation for your API:

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Testing the API

Once the server is running, you can test the API endpoints by visiting `http://127.0.0.1:8000/docs`, where you’ll find interactive documentation to execute requests.

Alternatively, use `curl` or a tool like **Postman** to test your API endpoints.

Example request with `curl`:

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/' \
  -H 'accept: application/json'
```

## Deployment

For production, it’s recommended to run Uvicorn with a process manager like **Gunicorn** and additional workers:

```bash
gunicorn -k uvicorn.workers.UvicornWorker app.main:app --workers 4
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```