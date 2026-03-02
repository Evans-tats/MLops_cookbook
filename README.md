## **Machine Learning Operations (MLOps) Project**

This repository contains a streamlined **MLOps pipeline** designed for high-performance model serving. It integrates automated testing, containerization via **Docker**, and continuous integration using **AWS CodeBuild**.

---

### 📁 **Project Components**

| File | Description |
| --- | --- |
| **`app.py`** | FastAPI entry point for the model microservice. |
| **`mlib.py`** | Core library containing model logic and helper functions. |
| **`cli.py`** | Command-line tool for local model interaction. |
| **`Makefile`** | Automation script for installation, linting, and testing. |
| **`Dockerfile`** | Container configuration for production deployment. |
| **`buildspec.yml`** | AWS CodeBuild instructions for CI/CD automation. |
| **`model.joblib`** | The serialized trained model ready for inference. |
| **`htwtmlb.csv`** | Reference dataset used within the application. |

---

### 🛠 **Getting Started**

This project uses a `Makefile` to ensure a consistent environment across different machines.

1. **Environment Setup**
```bash
python -m venv .venv
source .venv/bin/activate

```


2. **Install & Verify**
Install dependencies and run code quality checks (linting/testing):
```bash
make install
make lint
make test

```


3. **Local Execution**
Start the FastAPI server:
```bash
python app.py

```



---

### 🚀 **Deployment & Containerization**

#### **Docker**

To build and run the application as a standalone container:

```bash
# Build image
docker build -t ml-ops-service .

# Run container
docker run -p 8080:8080 ml-ops-service

```

#### **Continuous Integration**

The **`buildspec.yml`** file automates the build process in the cloud. Upon every push to the repository, the pipeline:

1. Installs the environment defined in `requirements.txt`.
2. Runs the `Makefile` test suite.
3. Prepares the artifact for deployment to AWS (ECR/Lambda/App Runner).

---

### 🔍 **Testing Inference**

Once the server is running, use the provided shell script to verify the prediction endpoint:

```bash
./predict.sh

```
