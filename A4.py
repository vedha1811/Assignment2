import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean, variance


file_path = r"/content/Lab Session Data.xlsx"
data = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")


price_column = data["Price"]
print(f"D = {price_column}")

price_mean = mean(price_column)
price_variance = variance(price_column)
print(f"The mean of column D is = {price_mean}")
print(f"The variance of column D is = {price_variance}")


data["Date"] = pd.to_datetime(data["Date"])
data["Weekday"] = data["Date"].dt.weekday

wednesday_prices = data[data["Weekday"] == 2]["Price"]
wednesday_mean = wednesday_prices.mean()
print(f"The sample mean for all Wednesdays in the dataset is = {wednesday_mean}")

data["Month"] = data["Date"].dt.month
april_prices = data[data["Month"] == 4]["Price"]
april_mean = mean(april_prices)
print(f"The sample mean for April in the dataset is = {april_mean}")


loss_probability = (data["Chg%"] < 0).mean()
print(f"The probability of making a loss in the stock is {loss_probability}")


profit_wednesdays = (data.loc[data["Weekday"] == 2, "Chg%"] > 0).mean()
print(f"The probability of making a profit in the stock on Wednesday is {profit_wednesdays}")


num_wed = len(wednesday_prices)
num_profitable_wed = (wednesday_prices > 0).sum()
conditional_prob_wed = num_profitable_wed / num_wed
print(f"The conditional probability of making a profit, given that today is Wednesday = {conditional_prob_wed}")

sns.scatterplot(x="Weekday", y="Chg%", data=data, hue="Weekday", palette="hls")
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("Chg% Distribution by Day of the Week")
plt.show()
