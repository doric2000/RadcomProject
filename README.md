# RADCOM Cyber Classifier

A machine learning project for network traffic classification using two distinct models: Application Classification and Attribution Classification.

## Project Overview

This project provides tools to train and deploy machine learning models that classify network traffic data. It includes a complete pipeline from data processing, model training, evaluation, and a deployable web application with Docker support.

## Components

### Machine Learning Models

#### app_model.py
The Application Classification model. This module:
- Loads training data from `data/APP-1/` directory
- Performs feature engineering including spectral analysis, inter-arrival features, flag densities, and balance features
- Trains an ensemble model using Random Forest, Gradient Boosting, and XGBoost as base classifiers with a Voting Classifier
- Saves trained model artifacts to the `models/` directory
- Generates submission predictions for validation data

To train and generate predictions:
```bash
python app_model.py
```

#### att_model.py
The Attribution Classification model. This module:
- Loads training data from `data/attribution/` directory
- Extracts isolation-based features including large packet detection, neighbor isolation, directionality analysis, and burst signals
- Trains an ensemble model using Random Forest and KNN with a Voting Classifier
- Saves trained model artifacts to the `models/` directory
- Generates submission predictions for validation data

To train and generate predictions:
```bash
python att_model.py
```

### Visualization

#### plots.py
Generates evaluation plots and visualizations for both models. This module:
- Creates confusion matrices for model performance evaluation
- Plots feature importance charts for tree-based models
- Generates protocol distribution and correlation heatmaps
- Saves all plots to `result/plots/` directory

To generate all plots:
```bash
python plots.py
```

### Utilities

#### log_setup.py
Configures logging for the project. Outputs logs to both stdout and a rotating log file (`run.log`).

### Web Application (app/ directory)

#### app/streamlit_app.py
The frontend web interface built with Streamlit. Allows users to:
- Select classification task type (app or att)
- Upload CSV validation files
- View data previews
- Run predictions and download results

#### app/api.py
FastAPI backend that:
- Receives prediction requests from the frontend
- Sends tasks to the Celery worker queue
- Returns prediction results to the frontend

#### app/tasks.py
Celery task definitions for asynchronous prediction processing. Loads trained models and processes prediction requests from the queue.

#### app/worker.py
Celery worker configuration for processing tasks from Redis queue.

## Data Structure

- `data/APP-1/` - Training, test, and validation data for Application Classification
- `data/attribution/` - Training, test, and validation data for Attribution Classification

## Running the Project

### Local Training

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the Application model:
```bash
python app_model.py
```

3. Train the Attribution model:
```bash
python att_model.py
```

4. Generate evaluation plots:
```bash
python plots.py
```

### Docker Deployment

The project includes Docker configuration for containerized deployment:

```bash
cd app
docker-compose up --build
```

This starts:
- Redis for task queue
- FastAPI backend for API endpoints
- Celery worker for ML predictions
- Streamlit frontend for user interface

Access the web interface at `http://localhost:8501` after deployment.

## Output Files

- `models/` - Trained model files (.pkl)
- `result/` - Submission CSV files with predictions
- `result/plots/` - Evaluation visualizations
- `run.log` - Execution logs
