FROM python:3.9-slim-buster
WORKDIR /app
COPY . /app
RUN apt update -y  && apt install awscli -y
RUN pip install -r requirements.txt && rm -rf /root/.cache/pip
CMD ["python", "app.py"]