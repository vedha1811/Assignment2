import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r"/content/Lab Session Data.xlsx"
data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

print(data.head())

print("\nData Types of Each Column:\n", data.dtypes)


categorical_columns = data.select_dtypes(include=["object"]).columns
print("\nCategorical Columns:\n", categorical_columns)

numerical_columns = data.select_dtypes(include=["number"]).columns
print("\nRange of Numeric Variables:\n", data[numerical_columns].agg(['min', 'max']))

missing_counts = data.isnull().sum()
print("\nMissing Values in Each Attribute:\n", missing_counts)


plt.figure(figsize=(12, 6))
sns.boxplot(data=data[numerical_columns])
plt.xticks(rotation=45)
plt.title("Boxplot to Detect Outliers")
plt.show()


numeric_means = data[numerical_columns].mean()
numeric_stddevs = data[numerical_columns].std()

print("\nMean of Numeric Variables:\n", numeric_means)
print("\nStandard Deviation of Numeric Variables:\n", numeric_stddevs)
