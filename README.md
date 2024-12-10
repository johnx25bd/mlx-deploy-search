# Search App, Containerized

This repository extends [mlx-two-tower-search](https://github.com/johnx25bd/mlx-two-tower-search) to include Docker Compose, user input logging, and finetuning. This is a fully containerized web application designed to demonstrate a robust MLOps and deployment setup. This project leverages FastAPI for the backend, Streamlit for the frontend, and Nginx as a reverse proxy, all orchestrated using Docker Compose.

A version of this app is deployed on [simplesearchengine.com](http://simplesearchengine.com), though we don't have SSL configured, so you may need to click through the warning.

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/johnx25bd/mlx-deploy-search.git
   cd mlx-deploy-search
   ```

2. **Build and start the containers:**
   ```bash
   docker compose up --build
   ```

3. **Access the applications:**
   - **Frontend (Streamlit):** `http://localhost/`
   - **API (FastAPI):** `http://localhost/api/`

(Note, however, this is an in-development version. The deployed code is hosted on [this repo](https://github.com/kalebsofer/DeploySearch).)

## Overview

This application showcases a modern architecture for deploying machine learning models and web applications. It is designed to be easily deployable and scalable, making it an ideal portfolio piece for demonstrating your skills in MLOps and containerization.

## Architecture

The application architecture is as follows:

- **Client Request** → **Nginx (Port 80)** → **FastAPI (Port 8000)**
- **Streamlit (Port 8501)**

### Components

- **Nginx**: Acts as a reverse proxy, routing client requests to the appropriate service.
- **FastAPI**: Serves as the backend API, handling data processing and model inference.
- **Streamlit**: Provides an interactive frontend for users to interact with the application.

## Prerequisites

To run this application, ensure you have the following installed:

- Docker and Docker Compose
- Git
- Python 3.12 (for local development)

## Project Structure

```
.
├── README.md
├── docker-compose.yml
├── api/
│   ├── Dockerfile
│   ├── app/
│   │   ├── main.py
│   │   └── ...
│   └── requirements.txt
├── streamlit/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
└── nginx/
    ├── Dockerfile
    └── conf/
        └── nginx.conf
```

## Development

### Local Development Setup

1. **Set up Python virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   cd api && pip install -r requirements.txt
   cd ../streamlit && pip install -r requirements.txt
   ```

3. **Run services locally:**
   - **FastAPI:**
     ```bash
     cd api
     uvicorn app.main:app --reload
     ```
   - **Streamlit:**
     ```bash
     cd streamlit
     streamlit run app.py
     ```

### Docker Commands

- **Build containers:**
  ```bash
  docker compose build
  ```

- **Start services:**
  ```bash
  docker compose up
  ```

- **Stop services:**
  ```bash
  docker compose down
  ```

## Deployment

1. **Push changes to repository**
2. **SSH into your server**
3. **Pull latest changes:**
   ```bash
   git pull
   ```
4. **Build and start containers:**
   ```bash
   docker compose up --build -d
   ```

## Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Docker Documentation](https://docs.docker.com/)

## Known Issues

- Streamlit may have connectivity issues in Brave browser due to privacy settings.
  - **Solution:** Disable Brave Shields for the site or use a different browser.

## Contributing

1. **Fork the repository**
2. **Create your feature branch (`git checkout -b feature/AmazingFeature`)**
3. **Commit your changes (`git commit -m 'Add some AmazingFeature'`)**
4. **Push to the branch (`git push origin feature/AmazingFeature`)**
5. **Open a Pull Request**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
