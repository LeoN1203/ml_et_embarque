FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7899

CMD ["fastapi", "run", "main.py", "--port", "7899"]