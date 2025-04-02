import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import itertools
import joblib  # For saving the model
import os
import seaborn as sns
import pandas as pd
import mlflow


class RidgePenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, W):
        # Calculate the Ridge Penalty (L2 Regularization)
        return self.l * np.sum(np.square(W))  # Regularization term

    def derivation(self, W):
        # Derivative of the Ridge Penalty
        return self.l * 2 * W  # Gradient of regularization term


class MyLogisticRegression:
    def __init__(self, k, n, lr=0.001, max_iter=1000, use_penalty=False, penalty=None, momentum=0.9):
        self.k = k
        self.n = n
        self.lr = lr
        self.max_iter = max_iter
        self.use_penalty = use_penalty
        self.penalty = penalty
        self.momentum = momentum
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_t = 0
        self.v_t = 0
        self.t = 0
        self.W = np.random.randn(self.n, self.k) * np.sqrt(2 / (self.n + self.k))

    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
    
    def precision(self, y_true, y_pred, c):
        TP = np.sum((y_pred == c) & (y_true == c))
        FP = np.sum((y_pred == c) & (y_true != c))
        return TP / (TP + FP) if (TP + FP) > 0 else 0
    
    def recall(self, y_true, y_pred, c):
        TP = np.sum((y_pred == c) & (y_true == c))
        FN = np.sum((y_pred != c) & (y_true == c))
        return TP / (TP + FN) if (TP + FN) > 0 else 0
    
    def f1_score(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def macro_average(self, metrics):
        return np.mean(metrics)
    
    def weighted_average(self, metrics, supports):
        return np.sum(np.array(metrics) * np.array(supports)) / np.sum(supports)
    
    def gradient(self, X, y):
        m = X.shape[0]
        H = self._predict(X, self.W)
        H = np.clip(H, 1e-15, 1 - 1e-15)
        loss = -np.sum(y * np.log(H)) / m
        grad = X.T @ (H - y)

        if self.use_penalty and self.penalty is not None:
            loss += self.penalty(self.W)
            grad += self.penalty.derivation(self.W)

        return loss, grad
    
    def evaluate(self, y_true, y_pred):
        classes = np.unique(y_true)
        supports = [np.sum(y_true == c) for c in classes]
        
        precisions = [self.precision(y_true, y_pred, c) for c in classes]
        recalls = [self.recall(y_true, y_pred, c) for c in classes]
        f1s = [self.f1_score(precisions[i], recalls[i]) for i in range(len(classes))]
        
        accuracy = self.accuracy(y_true, y_pred)
        
        macro_precision = self.macro_average(precisions)
        macro_recall = self.macro_average(recalls)
        macro_f1 = self.macro_average(f1s)
        
        weighted_precision = self.weighted_average(precisions, supports)
        weighted_recall = self.weighted_average(recalls, supports)
        weighted_f1 = self.weighted_average(f1s, supports)
        
        # Return all metrics as a dictionary
        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1
        }

    def fit(self, X_train_por, y_train_por, X_val, y_val):
        X_train_por = np.array(X_train_por)
        X_val = np.array(X_val)
        y_train_por = pd.get_dummies(y_train_por, dtype=int).to_numpy()

        self.losses = []
        batch_size = int(0.2 * X_train_por.shape[0])  
        evaluate_interval = 700

        model_path = "./models/"
        os.makedirs(model_path, exist_ok=True)

        # Log hyperparameters
        mlflow.log_param("learning_rate", self.lr)
        mlflow.log_param("max_iter", self.max_iter)
        mlflow.log_param("use_penalty", self.use_penalty)
        mlflow.log_param("momentum", self.momentum)
        mlflow.log_param("regularization", self.penalty.l if self.penalty else None)

        for i in range(self.max_iter):  
            ix = np.random.randint(0, X_train_por.shape[0], size=batch_size)
            batch_X = X_train_por[ix]
            batch_y = y_train_por[ix]
            loss, grad = self.gradient(batch_X, batch_y)

            if not np.isnan(loss):
                self.losses.append(loss)
            else:
                print(f"NaN loss detected at iteration {i}. Stopping training.")
                break

            # Adam Optimization
            self.t += 1
            self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * grad
            self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * (grad ** 2)
            m_t_hat = self.m_t / (1 - self.beta1 ** self.t)
            v_t_hat = self.v_t / (1 - self.beta2 ** self.t)
            self.W -= self.lr * m_t_hat / (np.sqrt(v_t_hat) + self.epsilon)

            # ✅ Evaluate and log only every N steps
            if i % evaluate_interval == 0:
                y_pred_val = self.predict(X_val)
                custom_metrics = self.evaluate(y_val, y_pred_val)

                print(f"[{i}] Validation Metrics:")
                print(f"  Accuracy            : {custom_metrics['accuracy']:.4f}")
                print(f"  Macro Precision     : {custom_metrics['macro_precision']:.4f}")
                print(f"  Macro Recall        : {custom_metrics['macro_recall']:.4f}")
                print(f"  Macro F1 Score      : {custom_metrics['macro_f1']:.4f}")
                print(f"  Weighted Precision  : {custom_metrics['weighted_precision']:.4f}")
                print(f"  Weighted Recall     : {custom_metrics['weighted_recall']:.4f}")
                print(f"  Weighted F1 Score   : {custom_metrics['weighted_f1']:.4f}")

                mlflow.log_metric("val_accuracy", custom_metrics["accuracy"], step=i)
                mlflow.log_metric("macro_precision", custom_metrics["macro_precision"], step=i)
                mlflow.log_metric("macro_recall", custom_metrics["macro_recall"], step=i)
                mlflow.log_metric("macro_f1", custom_metrics["macro_f1"], step=i)
                mlflow.log_metric("weighted_precision", custom_metrics["weighted_precision"], step=i)
                mlflow.log_metric("weighted_recall", custom_metrics["weighted_recall"], step=i)
                mlflow.log_metric("weighted_f1", custom_metrics["weighted_f1"], step=i)

            # ✅ Optional early stopping
            if len(self.losses) > 10:
                if abs(self.losses[-1] - self.losses[-2]) < 1e-6:
                    print(f"[EARLY STOP] at iteration {i}")
                    break



    def softmax(self, h_theta):
        exp_h = np.exp(h_theta - np.max(h_theta, axis=1, keepdims=True))
        return exp_h / np.sum(exp_h, axis=1, keepdims=True)

    def _predict(self, X, W):
        return self.softmax(X @ W)

    def predict(self, X_test):
        X_test = np.array(X_test)
        return np.argmax(self._predict(X_test, self.W), axis=1)
    
    def my_classification_report(self, y_test, y_pred):
        cols = ["precision", "recall", "f1-score"]
        idx = list(range(self.k)) + ["accuracy", "macro", "weighted"]

        report_data = []

        # Store entries for each class
        for c in range(self.k):
            p = self.precision(y_test, y_pred, c)
            r = self.recall(y_test, y_pred, c)
            f1 = self.f1_score(p, r)
            report_data.append([p, r, f1])

        # Accuracy
        accuracy = self.accuracy(y_test, y_pred)
        report_data.append([accuracy, accuracy, accuracy])  # ใช้ค่านี้แสดงเป็นแถวเดียว

        # Macro averages
        precisions = [row[0] for row in report_data[:self.k]]
        recalls = [row[1] for row in report_data[:self.k]]
        f1s = [row[2] for row in report_data[:self.k]]
        macro = [
            self.macro_average(precisions),
            self.macro_average(recalls),
            self.macro_average(f1s)
        ]
        report_data.append(macro)

        # Weighted averages
        supports = [np.sum(y_test == c) for c in range(self.k)]
        weighted = [
            self.weighted_average(precisions, supports),
            self.weighted_average(recalls, supports),
            self.weighted_average(f1s, supports)
        ]
        report_data.append(weighted)

        # สร้าง DataFrame เพื่อแสดงผล
        df = pd.DataFrame(report_data, index=idx, columns=cols)
        return df


    # def permutation_importance(estimator, X, y, scoring='accuracy', n_repeats=10, random_state=42):
    #     # ถ้า X เป็น DataFrame ให้แปลงเป็น numpy array ก่อน
    #     if hasattr(X, 'values'):
    #         X = X.values.copy()
    #     else:
    #         X = X.copy()
        
    #     def score_func(estimator, X, y):
    #         y_pred = estimator.predict(X)
    #         return np.sum(y == y_pred) / len(y)  # ใช้ accuracy

    #     # คำนวณ baseline score
    #     baseline_score = score_func(estimator, X, y)
    #     rng = np.random.RandomState(random_state)
        
    #     n_features = X.shape[1]
    #     importances = np.zeros(n_features)
    #     importances_std = np.zeros(n_features)
        
    #     # สำหรับแต่ละ feature
    #     for col in range(n_features):
    #         scores = []
    #         for _ in range(n_repeats):
    #             X_permuted = X.copy()
    #             rng.shuffle(X_permuted[:, col])  # สุ่มสับค่าในคอลัมน์นั้น
    #             score = score_func(estimator, X_permuted, y)
    #             scores.append(baseline_score - score)  # การลดลงของ accuracy
    #         importances[col] = np.mean(scores)
    #         importances_std[col] = np.std(scores)
        
    #     return {
    #         "importances_mean": importances,
    #         "importances_std": importances_std
    #     }

    # # สมมุติว่า X_test เป็น DataFrame ที่มีคอลัมน์ชื่อ feature ต่าง ๆ
    # # และ model คือ MyLogisticRegression ที่เทรนเรียบร้อยแล้ว
    # feature_names = X_test.columns

    # # เรียกใช้งาน permutation_importance ที่เราเขียนเอง
    # feature_importance = permutation_importance(
    #     estimator=model,
    #     X=X_test,
    #     y=y_test,
    #     scoring='accuracy',
    #     n_repeats=10,
    #     random_state=42
    # )

    # importances_mean = feature_importance["importances_mean"]
    # importances_std = feature_importance["importances_std"]

    # # เรียงลำดับจากมากไปน้อย
    # indices = np.argsort(importances_mean)[::-1]
    # sorted_importances_mean = importances_mean[indices]
    # sorted_importances_std = importances_std[indices]
    # sorted_feature_names = feature_names[indices]

    # # สร้างกราฟ bar chart แบบแนวนอน
    # plt.figure(figsize=(8, 5))
    # plt.barh(sorted_feature_names,
    #         sorted_importances_mean,
    #         xerr=sorted_importances_std,  # ถ้าอยากให้มี error bar
    #         color='skyblue', edgecolor='black')

    # plt.xlabel("Feature Importance")
    # plt.ylabel("Features")
    # plt.title("Feature Importance in Custom Logistic Regression")
    # plt.grid(axis="x", linestyle="--", alpha=0.7)
    # plt.gca().invert_yaxis()  # ให้ feature สำคัญสุดอยู่บนสุด
    # plt.show()
