import numpy as np
import matplotlib.pyplot as plt

# Sample data: two groups
np.random.seed(42)
group_a = np.random.normal(loc=10, scale=2, size=30)
group_b = np.random.normal(loc=11, scale=2, size=35) # Slightly higher mean

# Observed difference in means
observed_diff = np.mean(group_b) - np.mean(group_a)
print(f"Observed difference in means: {observed_diff:.2f}")

# Combine all data
combined_data = np.concatenate([group_a, group_b])
n_a = len(group_a)
n_b = len(group_b)
n_permutations = 10000
permutation_diffs = []

for _ in range(n_permutations):
    # Shuffle the combined data
    np.random.shuffle(combined_data)
    # Create two new groups
    perm_group_a = combined_data[:n_a]
    perm_group_b = combined_data[n_a:]
    # Calculate the difference in means for this permutation
    permutation_diffs.append(np.mean(perm_group_b) - np.mean(perm_group_a))

# Calculate p-value
p_value = np.sum(np.abs(permutation_diffs) >= np.abs(observed_diff)) / n_permutations
print(f"P-value from permutation test: {p_value:.4f}")

# Plotting the null distribution and observed difference
plt.figure(figsize=(10, 6))
plt.hist(permutation_diffs, bins=50, density=True, alpha=0.7, label='Permutation Differences')
plt.axvline(observed_diff, color='red', linestyle='dashed', linewidth=2, label=f'Observed Diff = {observed_diff:.2f}')
plt.title('Permutation Test for Difference in Means')
plt.xlabel('Difference in Means')
plt.ylabel('Density')
plt.legend()
plt.show()
