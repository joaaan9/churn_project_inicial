FROM python:3.7-slim-buster

ENV BACKEND postgresql://mlflow:mlflow@postgres_ml:5432/mlflow
ENV ARTIFACTS /mlruns

RUN apt-get update && apt-get install -y git
RUN pip install mlflow psycopg2-binary


ENTRYPOINT  mlflow server \
    --default-artifact-root $ARTIFACTS\
    --backend-store-uri $BACKEND \
    --host 0.0.0.0 \
    --port 5000