from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data with distinct clusters
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Initialize and fit GMM
n_components = 4 # We know there are 4 true clusters
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(X)

# Predict cluster assignments
labels = gmm.predict(X)

# Plot the results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', legend='full')
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

print("\nGMM Converged:", gmm.converged_)
print("GMM Weights:", gmm.weights_)
print("GMM Means:\n", gmm.means_)
print("GMM Covariances:\n", gmm.covariances_)
