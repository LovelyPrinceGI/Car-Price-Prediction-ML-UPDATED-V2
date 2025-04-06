print("=== LOADING TEST FILE ===")
# To run: ~/code# PYTHONPATH=. pytest tests/test_app_callbacks_v2.py -v
import pytest
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import warnings
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ✅ import model class ของเราเอง (ชื่อไฟล์กับ class ต้องตรง!)
from my_model.my_model import MyLogisticRegression  # <-- แก้ตรงนี้ตามจริง

# # ✅ MLflow model แยกตัวแปรต่างหาก
# mlflow.set_tracking_uri("https://admin:password@mlflow.ml.brain.cs.ait.ac.th")
# model_uri = "models:/st124876-a3-model/4"
# mlflow_model = mlflow.pyfunc.load_model(model_uri)  # อย่าใช้ชื่อว่า model ซ้ำ!


stage = "staging"

def load_mlflow(stage: str):
    # Rely on the environment variable set at runtime
    tracking_uri = os.environ.get("ML_FLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        raise ValueError("ML_FLOW_TRACKING_URI is not set in the environment.")
    # Construct the model URI, e.g., "models:/my-model/staging"
    model_uri = f"models:/st124876-a3-model/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

# from utlis import load_mlflow

def test_load_model():
    model = load_mlflow(stage)
    assert model


# ========== Test for Your MyLogisticRegression Model ==========

@pytest.mark.depends(on=['test_load_model']) 
def test_custom_model_accuracy():
    # model = MyLogisticRegression(k=4, n=5)
    model = load_mlflow(stage)
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 1, 3])
    accuracy = model.accuracy(y_true, y_pred)
    assert accuracy == 0.75, f"Expected accuracy 0.75, but got {accuracy}"

@pytest.mark.depends(on=['test_load_model']) 
def test_all_metrics_above_threshold():
    print("=== TEST: Evaluate All Custom Metrics ===")

    # model = MyLogisticRegression(n=7, k=3)
    model = load_mlflow(stage)
    model.W = np.random.rand(7, 3)  # Random weights for testing

    # Dummy validation set
    X_val = np.random.rand(20, 7)
    y_val = np.random.choice([0, 1, 2], size=20)

    y_pred = model.predict(X_val)
    metrics = model.evaluate(y_val, y_pred)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_precision"] <= 1
    assert 0 <= metrics["macro_recall"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert 0 <= metrics["weighted_precision"] <= 1
    assert 0 <= metrics["weighted_recall"] <= 1
    assert 0 <= metrics["weighted_f1"] <= 1

@pytest.mark.depends(on=['test_load_model']) 
def test_custom_model_training():
    # model = MyLogisticRegression(k=4, n=5, max_iter=10)
    model = load_mlflow(stage)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 4, 100)
    X_val = np.random.randn(20, 5)
    y_val = np.random.randint(0, 4, 20)
    
    model.fit(X_train, y_train, X_val, y_val)
    assert len(model.losses) > 0, "Training losses should not be empty."
    assert model.W is not None, "Model weights should not be None after training"
    assert model.W.shape == (5, 4), f"Expected W shape (5,4), but got {model.W.shape}"

@pytest.mark.depends(on=['test_load_model']) 
def test_custom_model_prediction():
    # model = MyLogisticRegression(k=4, n=5)
    model = load_mlflow(stage)
    X_test = np.random.randn(10, 5)
    y_pred = model.predict(X_test)
    assert y_pred.shape == (10,), f"Expected shape (10,), but got {y_pred.shape}"

# ========== Test for Your MLflow Loaded Model ==========

@pytest.mark.depends(on=['test_load_model']) 
def test_mlflow_model_output_shape():
    test_input = pd.DataFrame([[1, 2, 3, 4, 5, 6, 7]])  # ✅ ต้องตรงกับ feature count
    output = mlflow_model.predict(test_input)
    assert output.shape == (1,), f"Expecting shape (1,) but got {output.shape}"

@pytest.mark.depends(on=['test_load_model']) 
def test_mlflow_model_prediction():
    test_input = pd.DataFrame([[1, 2, 3, 4, 5, 6, 7]])
    output = mlflow_model.predict(test_input)
    assert isinstance(output, np.ndarray), "Model output should be a numpy array."

@pytest.mark.depends(on=['test_load_model']) 
def test_model_coeff_shape():
    # model = MyLogisticRegression(k=4, n=5)
    model = load_mlflow(stage)
    output = model.W
    assert output.shape == (5, 4), f"Expected shape (5, 4), but got {output.shape}"

@pytest.mark.depends(on=['test_load_model']) 
@pytest.fixture
def dummy_data():
    y_true = np.array([0, 1, 2, 1, 0, 2, 1, 2, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 0, 2, 2, 2, 1, 0])
    return y_true, y_pred

@pytest.mark.depends(on=['dummy_data']) 
def test_evaluate_metrics(dummy_data):
    # model = MyLogisticRegression(k=3, n=5)
    model = load_mlflow(stage)
    y_true, y_pred = dummy_data
    metrics = model.evaluate(y_true, y_pred)
    for key in [
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1"
    ]:
        assert key in metrics, f"{key} missing from output"
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range"

@pytest.mark.depends(on=['test_load_model']) 
def test_classification_report_shape(dummy_data):
    # model = MyLogisticRegression(k=3, n=5)
    model = load_mlflow(stage)
    y_true, y_pred = dummy_data
    report = model.my_classification_report(y_true, y_pred)
    assert isinstance(report, pd.DataFrame)
    assert report.shape == (6, 3)
    assert list(report.columns) == ["precision", "recall", "f1-score"]
    assert not report.isnull().values.any(), "NaN found in classification report"