# Weather Prediction Project

## Introduction

This project aims to predict the weather in Jena, Germany, using historical weather data. The model is trained on a dataset that includes weather information for the first part of 2024, with data recorded every 10 minutes. The core of the project involves using a Long Short-Term Memory (LSTM) model for weather prediction.

The system is divided into multiple services deployed in Kubernetes (Minikube) with Docker containers. These services include data preprocessing, model training, prediction generation, data storage, and a user interface for displaying predictions. The project uses FastAPI for building APIs, Postgres for storing weather data, and Streamlit for the UI.

## Project Structure

### 1. `predictor` Folder
This folder contains the code for the **predictor service**, which:
- Exposes FastAPI routes for model training and prediction.
- Uses LSTM (Long Short-Term Memory) model implemented with PyTorch for weather prediction.
- Input: A DataFrame with historical weather data.
- Output: A list of predictions (24 hours or 7 days).
- It also includes a Dockerfile to build the service image.

### 2. `data` Folder
This folder contains the `weather_data.csv` which is used for model training and making predictions.

### 3. `connector` Folder
The **connector service** connects the predictor and Postgres services and includes methods for:
- Uploading CSV data to Postgres.
- Training the model by retrieving data from Postgres and sending it to the predictor for training.
- Retrieving predictions for the next 24 hours and the next 7 days by querying Postgres.
- Managing Postgres data: adding a new row, deleting the last row, and clearing all data.
- This service also exposes a FastAPI interface to interact with the methods.
- It includes a Dockerfile to build the service image.

### 4. `ui` Folder
The **UI service** uses Streamlit to create a graphical interface for displaying weather predictions:
- Displays predictions for the next 24 hours and the next 7 days.
- Calls the connector service to get the latest predictions on each refresh.
- It also contains a Dockerfile for building the image.

### 5. `scheduler` Folder
The **scheduler service** ensures the weather predictions are kept up to date by:
- Calling the connector service’s training method every hour to ensure the model is trained with the latest data.
- Includes a Dockerfile to build the image.

### 6. `yaml_files` Folder
This folder contains Kubernetes YAML files that define the deployment and service configurations for each service:
- Each service (predictor, connector, ui) has its own deployment and service YAML.
- Additionally, a cron job YAML file (scheduler.yaml) is included for scheduling the training of the model at hourly intervals.

## Installation

### Prerequisites
Before getting started, ensure you have the following prerequisites installed:

1. Docker 24.0.7: Minikube may not be compatible with higher versions of Docker, so we recommend installing version 24.0.7

2. Minikube: Minikube provides a local Kubernetes cluster for development and testing purposes. Install Minikube by following the instructions appropriate for your operating system [here](https://minikube.sigs.k8s.io/docs/start/).

3. kubectl: kubectl is the Kubernetes command-line tool used to interact with the cluster. You can install it using the instructions [here](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/).

4. Start Minikube: Once Minikube and kubectl are installed, start Minikube by running the following command:
```bash
minikube start
```
5. (Optional) Start minikube dashboard with command:
```bash
minikube dashboard
```
6. Allow tunnel option: This command is needed to be able to acces cluster ports with local machine:
```bash
minikube tunnel
```

### Setup Process

#### Step 1: Build and Push Docker Images to Minikube

For each service, follow these steps:

1. **Predictor Service**:
    - Navigate to the `predictor` folder in your terminal.
    - Build the Docker image:  
      ```bash
      docker build -t predictor .
      ```
    - Push the image to Minikube:
      ```bash
      minikube image load predictor
      ```

2. **Connector Service**:
    - Navigate to the `connector` folder.
    - Build the Docker image:
      ```bash
      docker build -t connector .
      ```
    - Push the image to Minikube:
      ```bash
      minikube image load connector
      ```

3. **UI Service**:
    - Navigate to the `ui` folder.
    - Build the Docker image:
      ```bash
      docker build -t ui .
      ```
    - Push the image to Minikube:
      ```bash
      minikube image load ui
      ```

4. **Scheduler Service**:
    - Navigate to the `scheduler` folder.
    - Build the Docker image:
      ```bash
      docker build -t scheduler .
      ```
    - Push the image to Minikube:
      ```bash
      minikube image load scheduler
      ```

#### Step 2: Deploy Services to Kubernetes

1. Navigate to the `yaml_files` folder.
2. Deploy each service and cron job:
    ```bash
    kubectl apply -f postgres.yaml
    kubectl apply -f predictor-deployment.yaml
    kubectl apply -f predictor-svc.yaml
    kubectl apply -f connector-deployment.yaml
    kubectl apply -f connector-svc.yaml
    kubectl apply -f ui-deployment.yaml
    kubectl apply -f ui-svc.yaml
    kubectl apply -f scheduler.yaml
    ```

#### Step 3: Upload CSV and Train the Model

Once all the services are running and accessible, you can proceed with uploading the CSV and training the model.

1. **Access the Connector API**:
   - Open your browser and go to `http://localhost:8003/docs` to access the FastAPI documentation for the connector service.

2. **Upload the CSV**:
   - Use the **POST** method at `/upload-csv` to upload the `weather_data.csv` file from the `data` folder to the Postgres database.

3. **Train the Model**:
   - After uploading the CSV, use the **POST** method at `/predictor/train` to trigger the model training process. This will use the data from Postgres to train the LSTM model.

#### Step 4: Access the UI

- The UI can be accessed at http://localhost:8501.


