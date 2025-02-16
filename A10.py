import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Define the file path and read the dataset
file_path = r"/content/Lab Session Data.xlsx"
data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Select the first 20 rows for analysis
subset_data = data.iloc[:20]

# Identify binary columns (columns containing only 0s and 1s)
binary_columns = [
    col for col in data.columns
    if set(data[col].dropna().unique()).issubset({0, 1}) and pd.api.types.is_numeric_dtype(data[col])
]

# Extract binary and numeric matrices
binary_matrix = subset_data[binary_columns].to_numpy()
numeric_matrix = subset_data.select_dtypes(include=["number"]).to_numpy()

def compute_similarity_matrices(matrix):
    """Computes Jaccard and Simple Matching Coefficients (SMC) for a given binary matrix."""
    n = matrix.shape[0]
    jaccard_matrix = np.zeros((n, n))
    smc_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            both_one = np.sum((matrix[i] == 1) & (matrix[j] == 1))
            both_zero = np.sum((matrix[i] == 0) & (matrix[j] == 0))
            mismatch_10 = np.sum((matrix[i] == 1) & (matrix[j] == 0))
            mismatch_01 = np.sum((matrix[i] == 0) & (matrix[j] == 1))

            denominator_jaccard = both_one + mismatch_10 + mismatch_01
            denominator_smc = denominator_jaccard + both_zero

            jaccard_matrix[i, j] = both_one / denominator_jaccard if denominator_jaccard != 0 else 0
            smc_matrix[i, j] = (both_one + both_zero) / denominator_smc if denominator_smc != 0 else 0

    return jaccard_matrix, smc_matrix

# Compute similarity matrices
jc_matrix, smc_matrix = compute_similarity_matrices(binary_matrix)
cos_matrix = cosine_similarity(numeric_matrix)

# Plot heatmaps for the similarity matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

heatmap_params = {"annot": True, "cmap": "YlOrBr", "fmt": ".2f"}

sns.heatmap(jc_matrix, ax=axes[0], **heatmap_params)
axes[0].set_title("Jaccard Coefficient")

sns.heatmap(smc_matrix, ax=axes[1], **heatmap_params)
axes[1].set_title("Simple Matching Coefficient")

sns.heatmap(cos_matrix, ax=axes[2], **heatmap_params)
axes[2].set_title("Cosine Similarity")

plt.tight_layout()
plt.show()
