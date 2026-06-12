FROM python:3.10-slim

WORKDIR /app

COPY src/hello_docker.py /app/hello_docker.py

CMD ["python", "hello_docker.py"]