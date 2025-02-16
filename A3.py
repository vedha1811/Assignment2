import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


file_path = r"/content/Lab Session Data.xlsx"
data = pd.read_excel(file_path, sheet_name="Purchase data")


data["Customer Type"] = ["RICH" if amount > 200 else "POOR" for amount in data["Payment (Rs)"]]

features = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]]
target = data["Customer Type"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=42)


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)


predictions = knn_model.predict(X_test)


print("Classification Report")
print(classification_report(y_test, predictions))
