# A3 Car Price Prediction - st124876

## Project Description
A machine learning model to predict car prices using features such as brand, model, transmission, and mileage. This project demonstrates the use of FastAPI, Docker, MLflow, and GitHub Actions for deployment.

## How to Run Locally

```bash
docker compose up --build
```

## MLflow Tracking
- Logged on: https://mlflow.cs.ait.ac.th/

## Experiment name: st124876-a3
- Best model registered as: st124876-a3-model/4

## CI/CD
- GitHub Actions workflow:
    - build_test.yml: Runs unit tests and builds the Docker image.
    - test-model-staging.yml: Runs unit tests with pytest depends

## Deployment
- Deployment was partially complete. SSH issue occurred during deployment via GitHub Actions to AIT's ml2025 server. Manually tested on local environment.