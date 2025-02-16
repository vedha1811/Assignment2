import pandas as pd
import numpy as np


file_path = r"/content/Lab Session Data.xlsx"
data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


binary_vectors = data.iloc[:2]


binary_columns = [
    col for col in data.columns
    if set(data[col].dropna().unique()).issubset({0, 1})
]


binary_data = binary_vectors[binary_columns]


vec1, vec2 = binary_data.iloc[0].values, binary_data.iloc[1].values

both_ones = np.sum((vec1 == 1) & (vec2 == 1))
both_zeros = np.sum((vec1 == 0) & (vec2 == 0))
mismatch_10 = np.sum((vec1 == 1) & (vec2 == 0))
mismatch_01 = np.sum((vec1 == 0) & (vec2 == 1))

total_elements = both_ones + both_zeros + mismatch_10 + mismatch_01


jaccard_coefficient = both_ones / (both_ones + mismatch_10 + mismatch_01) if (both_ones + mismatch_10 + mismatch_01) != 0 else 0
simple_matching_coefficient = (both_ones + both_zeros) / total_elements if total_elements != 0 else 0


print("\nBinary Attributes Considered:", binary_columns)
print("\nJaccard Coefficient:", round(jaccard_coefficient, 4))
print("Simple Matching Coefficient:", round(simple_matching_coefficient, 4))

if jaccard_coefficient < simple_matching_coefficient:
    print("\n SMC is higher than JC as it considers both 1-1 and 0-0 matches.")
    print(" JC is useful when we only care about feature presence (1s).")
else:
    print("\n JC and SMC values are close, indicating feature vector similarity.")
