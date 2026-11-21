# Pet Breed Image Classifier

A deep learning project that classifies pet images into **37 breed categories** using a custom Convolutional Neural Network (CNN). The project uses the Oxford-IIIT Pet dataset and provides a Streamlit web application for image predictions.

## Project Overview

The goal of this project is to build an end-to-end image classification workflow, including:

- Dataset preparation
- Image preprocessing
- Custom CNN model development
- Model training
- Model evaluation
- Image prediction
- Streamlit application
- Docker deployment

## Dataset

This project uses the **Oxford-IIIT Pet Dataset**, which contains images of cats and dogs across 37 breed categories.

The dataset is stored locally and is not included in this GitHub repository.

Expected local dataset location:

```text
data/
└── oxford-iiit-pet/
    └── images/
```

Make sure the dataset is available in this location before running the application or training pipeline.

## Model

The project uses a custom Convolutional Neural Network (CNN) for image classification.

The trained model checkpoint is stored locally at:

```text
outputs/best_model.pth
```

Model weights are excluded from GitHub. You must have the trained checkpoint in the expected location to use the prediction application.

## Project Structure

```text
week1_image_classifier/
│
├── data/                    # Local dataset (not committed)
├── exercises/               # Practice exercises
├── notebooks/               # Jupyter notebooks
├── outputs/                 # Model checkpoints (not committed)
├── results/                 # Evaluation results
│
├── src/
│   ├── dataset.py            # Dataset loading and preprocessing
│   ├── model.py              # Custom CNN architecture
│   ├── train.py              # Model training
│   ├── evaluate.py           # Model evaluation
│   ├── predict.py            # Prediction functionality
│   └── app.py                # Streamlit application
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rabiaasif294/week1_image_classifier.git
cd week1_image_classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```cmd
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the local files

- Place the Oxford-IIIT Pet dataset in the expected `data/` directory.
- Place the trained model checkpoint at `outputs/best_model.pth`.

These files are not downloaded automatically by the application.

## Training and Evaluation

Run the training script:

```bash
python src/train.py
```

Run the evaluation script:

```bash
python src/evaluate.py
```

Check the project scripts for any required command-line arguments or configuration before running them.

## Run the Streamlit Application Locally

From the project root, activate your virtual environment and run:

```bash
streamlit run src/app.py --server.port 8502
```

Open the application in your browser:

```text
http://localhost:8502
```

The application uses the local dataset and trained model checkpoint.

## Docker Deployment

Docker allows the application to run in an isolated environment with its own installed dependencies.

### 1. Build the Docker image

Run from the project root:

```bash
docker build -t pet-breed-classifier .
```

### 2. Run the container

On Windows CMD:

```cmd
docker run -d --name pet-breed-app -p 8502:8502 -v "E:\week1_image_classifier\data:/app/data:ro" -v "E:\week1_image_classifier\outputs:/app/outputs:ro" pet-breed-classifier
```

The dataset and model directories are mounted as read-only volumes. They remain on the host machine and are not embedded in the Docker image.

### 3. Open the Dockerized application

```text
http://localhost:8502
```

### 4. Check the container

```bash
docker ps
```

View application logs:

```bash
docker logs pet-breed-app
```

To stop the container:

```bash
docker stop pet-breed-app
```

To remove the stopped container:

```bash
docker rm pet-breed-app
```

## Important Notes

- The dataset and trained model checkpoint are stored locally.
- Do not commit dataset images, model weights, virtual environments, or secrets.
- The Docker image contains the application code and Docker dependencies.
- The dataset and checkpoint must be present on the host machine when running the Docker container.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow
- Streamlit
- Docker
- Git and GitHub

## Repository

[GitHub – week1_image_classifier](https://github.com/rabiaasif294/week1_image_classifier)