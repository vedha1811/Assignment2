import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define the file path
file_path = r"/content/Lab Session Data.xlsx"

# Load the Excel file and select the relevant sheet
data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Select only numeric columns
numeric_columns = data.select_dtypes(include=["number"]).columns
numeric_vectors = data[numeric_columns]

# Ensure at least two rows exist for comparison
if len(numeric_vectors) < 2:
    print("Error: Not enough data to compute cosine similarity.")
else:
    # Extract the first two numeric vectors
    vec1 = numeric_vectors.iloc[0].values.reshape(1, -1)
    vec2 = numeric_vectors.iloc[1].values.reshape(1, -1)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(vec1, vec2)[0][0]

    # Display results
    print("\nCosine Similarity between the first two observations:", round(cosine_sim, 4))

    if cosine_sim > 0.8:
        print("The vectors are highly similar.")
    elif cosine_sim > 0.5:
        print("The vectors have moderate similarity.")
    else:
        print("The vectors are not very similar.")
