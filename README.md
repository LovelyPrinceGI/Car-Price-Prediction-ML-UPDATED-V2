### ✅ **1. What your code is doing (Overview)**
You have implemented a Logistic Regression model **from scratch** with the following features:
- **Gradient Descent Optimization** using **Adam Optimizer**.
- **L2 Regularization (Ridge Penalty)** support.
- **Grid Search** for hyperparameter tuning.
- **MLFlow Integration** for logging parameters, metrics, and model files.

### 📌 **2. Class Definition**
- You defined two classes: `RidgePenalty` (for L2 Regularization) and `LogisticRegression` (the main model).

---

### 🔍 **RidgePenalty Class (L2 Regularization)**
```python
class RidgePenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        return self.l * 2 * theta
```
- This class applies L2 regularization which penalizes large weights by adding a term \( \lambda \sum \theta^2 \) to the loss function.
- `derivation()` returns the gradient of the regularization term, which helps in updating weights during training.

---

### 🔍 **LogisticRegression Class (Main Model)**
#### **Initialization (`__init__()`)**
- `k`: Number of classes.
- `n`: Number of features.
- `lr`: Learning rate.
- `max_iter`: Number of training iterations.
- `use_penalty`: Whether to use regularization or not.
- `penalty`: Type of regularization (if any).
- `momentum`: Momentum coefficient for Adam Optimizer.
- **Weight Initialization**: Weights are initialized using Xavier initialization.

---

#### **Training (`fit()`)**
```python
def fit(self, X, Y, X_val, Y_val)
```
- Uses **Adam Optimizer** for gradient updates.
- Logs metrics and parameters to **MLFlow**.
- Trains over a set number of iterations (`max_iter`).

---

#### **Gradient Calculation (`gradient()`)**
```python
def gradient(self, X, Y)
```
- Calculates the loss using the **Cross-Entropy Loss Function**.
- Computes gradients for updating the weights.
- Applies regularization if enabled.

---

#### **Prediction (`predict()`)**
```python
def predict(self, X_test)
```
- Uses **Softmax Activation Function** to predict class probabilities.
- Returns the class with the highest probability for each input.

---

#### **Grid Search (`grid_search()`)**
```python
def grid_search(self, X_train, Y_train_enc, X_val, Y_val_enc, param_grid)
```
- Loops over all parameter combinations specified in `param_grid`.
- Trains a new model for each combination.
- Logs the best parameters based on validation loss.

---

### 📈 **3. Logging with MLFlow**
You are logging:
- Hyperparameters: `learning_rate`, `max_iter`, `momentum`, `regularization`.
- Validation Metrics: `val_accuracy`, `val_precision`, `val_recall`, `val_f1`.
- Model files are saved and logged as artifacts.

---

### 🔥 **4. How to Visualize Results in MLFlow**
1. **Go to your MLFlow UI** (http://127.0.0.1:5000).
2. Click on your experiment (`Car_Price_Prediction_Class2`).
3. Compare different runs to see which parameters give the best results.
4. Visualize the training loss trend by looking at the `loss` metric if you decide to log it.

---

### 📊 **5. Recommendations to Improve Your Code**
1. **Log Training Loss:** 
   - You are currently logging only the final validation metrics. Logging the `loss` at each step will help you visualize the learning process.
2. **Add Plotting Feature:** 
   - Plot `self.losses` to see how well your model is converging during training.
3. **Improve `grid_search()`:**
   - Log each run's results with MLFlow to compare them easily.

---

