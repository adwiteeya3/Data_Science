from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Using the same X_train, X_test, y_train, y_test from the GLM example
# But let's assume y is continuous for regression
np.random.seed(42)
y_reg = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(100) * 5 # Continuous target

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    df[['feature1', 'feature2', 'feature3']], y_reg, test_size=0.3, random_state=42
)

# Initialize and train Lasso Regression
lasso_model = Lasso(alpha=0.1, random_state=42) # alpha is the regularization strength
lasso_model.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_lasso = lasso_model.predict(X_test_reg)

# Evaluate
print("\nLasso Regression Performance:")
print(f"Coefficients: {lasso_model.coef_}") # Some coeffs might be zeroed out
print(f"Mean Squared Error: {mean_squared_error(y_test_reg, y_pred_lasso):.2f}")
