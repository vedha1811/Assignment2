import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

file_path = r"/content/Lab Session Data.xlsx"
data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


numeric_columns = data.select_dtypes(include=["number"]).columns


plt.figure(figsize=(12, 6))
data[numeric_columns].hist(figsize=(12, 8), bins=20)
plt.suptitle("Before Normalization: Distribution of Numeric Attributes")
plt.show()

minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

normalized_data = data.copy()

for col in numeric_columns:
    if abs(data[col].skew()) < 1:  # Low skewness → Standard Scaling
        normalized_data[col] = standard_scaler.fit_transform(data[[col]])
    else:  # High skewness → Min-Max Scaling
        normalized_data[col] = minmax_scaler.fit_transform(data[[col]])

print("\nNormalization Completed!")


plt.figure(figsize=(12, 6))
normalized_data[numeric_columns].hist(figsize=(12, 8), bins=20)
plt.suptitle("After Normalization: Distribution of Numeric Attributes")
plt.show()
