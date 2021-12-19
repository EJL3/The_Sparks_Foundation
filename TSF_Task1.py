# Importing the required libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Reading the data

data = {
        "Hours": [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.8, 6.9, 7.8],
        "Scores": [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30, 24, 67, 69, 30, 54, 35, 76, 86]
}

df = pd.DataFrame(data)

# Plotting the data

df.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# Splitting the data into training and testing sets(Supervised Learning)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Model Results..")

line = regressor.coef_ * X + regressor.intercept_
plt.scatter(X, y)
plt.plot(X, line)
plt.show()

print(X_test)
y_pred = regressor.predict(X_test)
sf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred},)
print(sf)

hours = [[9.25]]
own_pred = regressor.predict(hours)
print("Number of hours = {}".format(hours))
print("Prediction Score = {}".format(own_pred[0]))