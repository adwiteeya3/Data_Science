from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sns

# Load example data (digits dataset)
digits = load_digits(n_class=6)
X, y = digits.data, digits.target

# Reduce dimensions using t-SNE
# For larger datasets, consider Barnes-Hut t-SNE (default for n_samples > 5000)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# Create a DataFrame for plotting
tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df['digit'] = y

# Plot the results
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="TSNE1", y="TSNE2",
    hue="digit",
    palette=sns.color_palette("hsv", len(np.unique(y))),
    data=tsne_df,
    legend="full",
    alpha=0.8
)
plt.title('t-SNE Visualization of Digits Dataset')
plt.show()
