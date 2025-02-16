import pandas as pd
import numpy as np


file_path = r"/content/Lab Session Data.xlsx"
data = pd.read_excel(file_path, sheet_name="Purchase data")


data = data.iloc[:, :5].drop(columns=["Customer"])
data.columns = ["Candies", "Mangoes", "Milk_Packets", "Payment"]


X = np.column_stack((np.ones(len(data)), data[["Candies", "Mangoes", "Milk_Packets"]].values))
Y = data["Payment"].values

X_pinv = np.linalg.pinv(X)
model_parameters = X_pinv @ Y


print("Estimated Model Parameters (Intercept and Coefficients):")
print(model_parameters)
