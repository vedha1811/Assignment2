import numpy as np
import pandas as pd

# Load dataset
data = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")

# Extract relevant columns
A = data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = data[['Payment (Rs)']].values

# Compute properties
dim = A.shape[1]
num_vectors = A.shape[0]
rank_A = np.linalg.matrix_rank(A)
A_pinv = np.linalg.pinv(A)
cost_vector = A_pinv @ C

# Display results
print(f"Matrix A:\n{A}")
print(f"Cost Vector C:\n{C}")
print(f"Dimensionality of vector space: {dim}")
print(f"Number of vectors in vector space: {num_vectors}")
print(f"Rank of matrix A: {rank_A}")
print(f"Pseudo-inverse of matrix A:\n{A_pinv}")
print(f"Estimated cost of each product:\n{cost_vector.flatten()}")
