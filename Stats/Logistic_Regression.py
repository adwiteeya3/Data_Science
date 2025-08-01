import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Sample Data Generation
np.random.seed(42)
X = np.random.rand(100, 3) * 10
y = (X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2] + np.random.randn(100) * 2 > 7).astype(int)

df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
df['target'] = y

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[['feature1', 'feature2', 'feature3']], df['target'], test_size=0.3, random_state=42
)

# Initialize and train GLM (Logistic Regression in this case)
# For a more general GLM (e.g., Poisson), you'd use statsmodels
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print("Logistic Regression (GLM) Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example with statsmodels for a true GLM (e.g., Poisson)
import statsmodels.api as sm

# Assuming 'counts' is your dependent variable (e.g., number of events)
# and 'features' are your independent variables
# For this example, let's create some dummy count data
np.random.seed(42)
counts = np.random.poisson(lam=5, size=100) + (X[:, 0] * 0.5).astype(int) # Simulated counts
df['counts'] = counts

# Prepare data for statsmodels
X_sm = sm.add_constant(df[['feature1', 'feature2', 'feature3']]) # Add intercept
y_sm = df['counts']

# Fit Poisson GLM
poisson_glm = sm.GLM(y_sm, X_sm, family=sm.families.Poisson())
poisson_results = poisson_glm.fit()

print("\nPoisson GLM (Statsmodels) Summary:")
print(poisson_results.summary())
