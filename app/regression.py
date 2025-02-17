import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class LinearRegression:
    # Split into 
    kfold = KFold(n_splits=3)

    def __init__(self, learning_rate=0.01, epochs=1000, weight_init='zeros', momentum=0.0, lambda_=0.1, 
                 regularization=None, function='linear', degree=2, batch_type='batch', batch_size=32):
        """
        batch_type: 'batch' (full dataset), 'mini-batch' (small batch), 'stochastic' (one sample at a time)
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_init = weight_init
        self.momentum = momentum
        self.lambda_ = lambda_
        self.regularization = regularization
        self.function = function  # 'linear' or 'poly'
        self.degree = degree  # Polynomial degree
        self.batch_type = batch_type
        self.batch_size = batch_size
        self.theta = None
        self.bias = None
        self.prev_step = 0

    def get_params(self):
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "weight_init": self.weight_init,
            "momentum": self.momentum,
            "lambda_": self.lambda_,
            "regularization": self.regularization.__class__.__name__ if self.regularization else None,
            "function": self.function,
            "degree": self.degree,
            "batch_type": self.batch_type,
            "batch_size": self.batch_size if self.batch_type == 'mini-batch' else None
        }

    def initialize_weights(self, n_features):
        """Initialize weights using Xavier or Zeros method."""
        if self.weight_init == 'xavier':
            limit = 1.0 / np.sqrt(n_features)
            self.theta = np.random.uniform(-limit, limit, size=(n_features,))
        else:
            self.theta = np.zeros(n_features)
        self.bias = 0

    def fit(self, x, y):
        """Train the model using Gradient Descent with optional Momentum and Cross-Validation."""
        if self.function == 'poly':
            x = self._transform_features(x)  # Transform input for polynomial regression

        n_samples, n_features = x.shape
        self.initialize_weights(n_features)

        # Ensure any active MLflow run is closed before starting a new one
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name="A2-LinearRegression Training"):
            mlflow.log_params(self.get_params())
            run = mlflow.active_run()
            prev_loss = float('inf')  # Initialize previous loss for early stopping

            for fold, (train_idx, val_idx) in enumerate(self.kfold.split(x)):
                x_train_sub, x_val = x.iloc[train_idx], x.iloc[val_idx]
                y_train_sub, y_val = y.iloc[train_idx], y.iloc[val_idx]

                for epoch in range(self.epochs):
                    if self.batch_type == 'batch':
                        # 🟢 Batch Gradient Descent
                        batch_x, batch_y = x_train_sub, y_train_sub
                    elif self.batch_type == 'stochastic':
                        # 🔴 Stochastic Gradient Descent (Random Single Sample)
                        random_idx = np.random.randint(len(x_train_sub))  
                        batch_x, batch_y = x_train_sub.iloc[random_idx:random_idx+1], y_train_sub.iloc[random_idx:random_idx+1]
                    else:
                        # 🔵 Mini-Batch Gradient Descent
                        actual_batch_size = min(self.batch_size, x_train_sub.shape[0])  # Ensure batch size is not larger than available samples
                        batch_indices = np.random.choice(x_train_sub.shape[0], actual_batch_size, replace=False)
                        batch_x, batch_y = x_train_sub.iloc[batch_indices], y_train_sub.iloc[batch_indices]

                    y_pred = np.dot(batch_x, self.theta) + self.bias
                    error = y_pred - batch_y

                    d_theta = (1 / len(batch_x)) * np.dot(batch_x.T, error)  
                    d_bias = (1 / len(batch_x)) * np.sum(error)  

                    if self.regularization:
                        d_theta += self.regularization.gradient(self.theta)

                    step = self.learning_rate * d_theta
                    self.theta = self.theta - step + self.momentum * self.prev_step
                    self.bias -= self.learning_rate * d_bias
                    self.prev_step = step

                    mse_loss = self.mse(y_pred, batch_y)

                    # ✅ Prevent NaN or Infinite Values in prev_loss
                    if np.isnan(mse_loss) or np.isinf(mse_loss):  
                        break
                    if epoch % 100 == 0:
                        mlflow.log_metric(f"train_mse_fold_{fold}", mse_loss, step=epoch)

                    if abs(prev_loss - mse_loss) < 1e-7:
                        break
                    prev_loss = mse_loss

                y_val_pred = self.predict(x_val)
                val_mse = self.mse(y_val_pred, y_val)
                mlflow.log_metric(f"val_mse_fold_{fold}", val_mse)

            mlflow.log_metric("final_bias", self.bias)
            mlflow.end_run()
            return run.info.run_id

    def predict(self, x):
        if self.function == 'poly':
            x = self._transform_features(x)  # Transform features before prediction
        return np.dot(x, self.theta) + self.bias

    def mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2) if not np.isnan(y_pred).any() else float('inf')  

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    def _transform_features(self, x):
        """Transform input features for polynomial regression."""
        return x ** self.degree
    
# Standard Linear Regression (No Regularization)
class NormalRegression(LinearRegression):
    def __init__(self, **kwargs):
        super().__init__(regularization=None, **kwargs)

# Lasso Regression: Uses L1 regularization
class LassoRegression(LinearRegression):
    def __init__(self, **kwargs):
        super().__init__(regularization=LassoPenalty(kwargs.get("lambda_", 0.1)), **kwargs)

# Ridge Regression: Uses L2 regularization
class RidgeRegression(LinearRegression):
    def __init__(self, **kwargs):
        super().__init__(regularization=RidgePenalty(kwargs.get("lambda_", 0.1)), **kwargs)

# ElasticNet Regression: Mixes L1 and L2 penalties
class ElasticNetRegression(LinearRegression):
    def __init__(self, l_ratio=0.5, **kwargs):
        super().__init__(regularization=ElasticNetPenalty(kwargs.get("lambda_", 0.1), l_ratio), **kwargs)


class LassoPenalty:
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def __call__(self, theta):
        return self.lambda_ * np.sum(np.abs(theta))

    def gradient(self, theta):
        return self.lambda_ * np.sign(theta)


class RidgePenalty:
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def __call__(self, theta):
        return self.lambda_ * np.sum(theta ** 2)

    def gradient(self, theta):
        return self.lambda_ * 2 * theta


class ElasticNetPenalty:
    def __init__(self, lambda_=0.1, l_ratio=0.5):
        self.lambda_ = lambda_
        self.l_ratio = l_ratio

    def __call__(self, theta):
        l1 = self.l_ratio * self.lambda_ * np.sum(np.abs(theta))  # L1
        l2 = (1 - self.l_ratio) * self.lambda_ * 0.5 * np.sum(theta ** 2)  # L2
        return l1 + l2

    def gradient(self, theta):
        l1_grad = self.lambda_ * self.l_ratio * np.sign(theta)
        l2_grad = self.lambda_ * (1 - self.l_ratio) * theta
        return l1_grad + l2_grad
