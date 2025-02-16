import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = r"/content/Lab Session Data.xlsx"
data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

print("\nMissing Values Before Imputation:\n", data.isnull().sum())

numeric_columns = data.select_dtypes(include=["number"]).columns
categorical_columns = data.select_dtypes(include=["object"]).columns

print("\nNumeric Columns:", numeric_columns)
print("\nCategorical Columns:", categorical_columns)


plt.figure(figsize=(12, 6))
sns.boxplot(data=data[numeric_columns])
plt.xticks(rotation=45)
plt.title("Boxplot to Detect Outliers")
plt.show()

data[numeric_columns] = data[numeric_columns].apply(lambda x: x.fillna(x.median() if x.skew() > 1 else x.mean()))


print("\nMissing Values After Numeric Imputation:\n", data.isnull().sum())

data[categorical_columns] = data[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]))


print("\nMissing Values After Categorical Imputation:\n", data.isnull().sum())

data.to_excel("Imputed_Thyroid_Data.xlsx", index=False)
print("Data imputation complete. File saved as 'Imputed_Thyroid_Data.xlsx'")
