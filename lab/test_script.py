import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import metrics


# Generate some random data
y_true = np.random.randint(2, size=100)
y_pred = np.random.randint(2, size=100)

# Calculate metrics using our implementations
acc_custom = metrics.accuracy(y_pred, y_true)
prec_custom = metrics.precision(y_pred, y_true)
rec_custom = metrics.recall(y_pred, y_true)
f1_custom = metrics.f1_score_metric(y_pred, y_true)

# Calculate metrics using scikit-learn
acc_sklearn = accuracy_score(y_true, y_pred)
prec_sklearn = precision_score(y_true, y_pred)
rec_sklearn = recall_score(y_true, y_pred)
f1_sklearn = f1_score(y_true, y_pred)

# Compare the results
print("Accuracy: Custom =", acc_custom, ", Scikit-learn =", acc_sklearn)
print("Precision: Custom =", prec_custom, ", Scikit-learn =", prec_sklearn)
print("Recall: Custom =", rec_custom, ", Scikit-learn =", rec_sklearn)
print("F1-score: Custom =", f1_custom, ", Scikit-learn =", f1_sklearn)

# Generate some random data
y_true = np.random.randn(100)
y_pred = y_true + np.random.randn(100) * 0.1

# Calculate metrics using our implementations
mse_custom = metrics.mse(y_pred, y_true)
rmse_custom = metrics.rmse(y_pred, y_true)
mae_custom = metrics.mae(y_pred, y_true)
r2_custom = metrics.r2_score(y_pred, y_true)

# Calculate metrics using scikit-learn
mse_sklearn = mean_squared_error(y_true, y_pred)
rmse_sklearn = np.sqrt(mean_squared_error(y_true, y_pred))
mae_sklearn = mean_absolute_error(y_true, y_pred)
r2_sklearn = r2_score(y_true, y_pred)

# Compare the results
print("MSE: Custom =", mse_custom, ", Scikit-learn =", mse_sklearn)
print("RMSE: Custom =", rmse_custom, ", Scikit-learn =", rmse_sklearn)
print("MAE: Custom =", mae_custom, ", Scikit-learn =", mae_sklearn)
print("R-squared: Custom =", r2_custom, ", Scikit-learn =", r2_sklearn)
