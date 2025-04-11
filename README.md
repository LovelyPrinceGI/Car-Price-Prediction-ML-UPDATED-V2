# **A3 Car Price Prediction - st124876**

## **Project Description**
This project redefines car price prediction as a **multi-class classification** problem by categorizing `selling_price` into 4 classes (e.g., low, medium-low, medium-high, high). We use a **custom logistic regression model** implemented from scratch to predict the class.

## **Key highlights:**

- Bucketed classification of selling_price

- Custom LogisticRegression class

- Training logs to MLflow every 700 iterations

- Manual implementation of classification metrics (accuracy, precision, recall, F1)

- Ridge (L2) regularization support

- Optimized using Adam Optimizer

- Full deployment using FastAPI and Docker

- CI/CD integration via GitHub Actions

The model logs training progress to **MLflow every 700 iterations**, allowing transparent experiment tracking. All classification metrics (accuracy, precision, recall, f1-score) are computed manually and validated against scikit-learn.

This project also demonstrates modern MLOps best practices including:
- Custom preprocessing
- MLflow experiment tracking
- Ridge (L2) regularization support
- Dockerized deployment using FastAPI
- CI/CD via GitHub Actions

---

## **🧪 Model Features**

- **Classification** based on selling price brackets

- **Custom Logistic Regression** class with:
  - Accuracy, Precision, Recall, F1-Score (per-class, macro, weighted)
  - Ridge Regularization
- **Manual metric calculations** for deeper understanding
- **Experiment Logging** to MLflow every 700 iterations

---

## **🔍 Dataset Preprocessing**

- Cleaned and updated dataset from `cars_updated.csv`
- Bucketed `selling_price` using `pd.cut()` to 4 classes
- Encoded categorical features and normalized numerical features

## **Model Training**

- Defined all functions manually for educational purposes
- Includes forward pass, backward pass, gradient computation, and metric evaluations
- Custom classification metric functions: accuracy, precision, recall, f1, macro, weighted
- Optionally includes Ridge regularization (L2)
- Experiment logs saved every 700 iterations via mlflow.log_metrics()
- Optimizer: Adam

The model is trained using the Adam optimization algorithm, which combines momentum and adaptive learning rate techniques for faster and more stable convergence.

## **💡 How Adam Works:**

- Maintains moving averages of gradients (m_t) and squared gradients (v_t)
- Applies bias correction for early steps (m_t_hat, v_t_hat)
- Updates weights using:

```bash
W -= learning_rate * m_t_hat / (sqrt(v_t_hat) + epsilon)
```

### **✅ Pros (for car price data):**

- Efficient with sparse and noisy features like fuel_efficiency, km_driven, etc.
- Fast convergence is useful due to non-convexity from many categorical encodings
- Handles scaling differences between input features well (e.g., power vs. year)

### **⚠️ Cons:**

- Might overshoot or oscillate near sharp minima
- May require fine-tuning of learning rate and β parameters


---

## **🚀 How to Run Locally**

```bash
docker compose up --build
```

## **🔎 MLflow Tracking**
```bash
Tracking URI: https://mlflow.cs.ait.ac.th/
```
- Experiment name: st124876-a3
- Model logged every 700 iterations
- Best model registered as: st124876-a3-model/4

## **🛠 CI/CD Pipeline**
GitHub Actions Workflows:
```bash
build_test.yml: Runs unit tests and builds Docker image
```
```
test-model-staging.yml: Tests model logic (input/output shape)
```
Auto-deploys upon passing unit tests

## **🌐 Deployment Status**
- Deployment fully completed
- Successfully tested locally via FastAPI server

## **📂 Folder Structure**

```bash
.
├── app/                      # FastAPI app
├── datasets/                # Original and updated dataset
├── preprocess_v2/           # Encoders and scalers
├── source_code/             # Jupyter notebooks and preprocessed data
├── my_model/                # Custom model implementation
├── mlruns/                  # MLflow run tracking
├── models/                  # Saved models
├── templates/               # index.html for frontend
├── static/                  # CSS styling
├── tests/                   # Unit tests for model and app
├── .github/workflows/       # CI/CD configuration
└── README.md
```


## **Example of predicting website UI**
![alt text](image-1.png)

## **🤝 Contributors (Special thanks)**
- Chaklam Silpasuwanchai
- Akraradet Sinsamersuk