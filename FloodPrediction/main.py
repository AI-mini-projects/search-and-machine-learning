import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def Scale(data):
    scaled_data = data.copy()
    for column in data.columns[1:-1]:
        col_min = data[column].min()
        col_max = data[column].max()
        scaled_data[column] = (data[column] - col_min) / (col_max - col_min)
    return scaled_data

def SGD(train_data, learning_rate, epoch, k, divideby):
    w = np.zeros(shape=(1, train_data.shape[1] - 2))
    b = 0
    current_iter = 1
    losses = []

    while current_iter <= epoch:
        temp = train_data.sample(k)
        y = np.array(temp['FloodProbability'])
        x = np.array(temp.drop(['FloodProbability', 'id'], axis=1))

        w_gradient = np.zeros_like(w)
        b_gradient = 0
        loss = 0

        for i in range(k):
            prediction = np.dot(w, x[i]) + b
            error = y[i] - prediction
            w_gradient += (-2) * x[i] * error
            b_gradient += (-2) * error
            loss += error ** 2
            losses.append(loss)

        w -= learning_rate * (w_gradient / k)
        b -= learning_rate * (b_gradient / k)

        learning_rate /= divideby
        current_iter += 1

    return w, b, losses


def predict(x, w, b):
    y_pred = []
    for i in range(len(x)):
        y = (np.dot(w, x[i]) + b).item()
        y_pred.append(y)
    return np.array(y_pred)

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

for column in train_data.columns:
    if train_data[column].isnull().sum() > 0:
        train_data[column].fillna(train_data[column].mean(), inplace=True)

for column in test_data.columns:
    if test_data[column].isnull().sum() > 0:
        test_data[column].fillna(test_data[column].mean(), inplace=True)

train_data = Scale(train_data)
test_data = Scale(test_data)
x_train = train_data.drop(['FloodProbability', 'id'], axis=1).values
y_train = train_data['FloodProbability'].values
x_test = test_data.drop(['FloodProbability', 'id'], axis=1).values
y_test = test_data['FloodProbability'].values

train_data['FloodProbability'] = y_train

learning_rate = 0.06
epoch = 1500
k = 32
divideby = 1

w, b, losses = SGD(train_data, learning_rate, epoch, k, divideby)

plt.plot(range(len(losses)), losses, label='Training Loss based on iteration')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Function (iterations)')
plt.legend()
plt.show()
epoch_loss = []
for i in range(0, len(losses), k):
    chunk = losses[i:i + k]
    epoch_loss.append(sum(chunk) / len(chunk))
plt.plot(range(len(epoch_loss)), epoch_loss, label='Training Loss based on epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function (epochs)')
plt.legend()
plt.show()
y_train_pred = predict(x_train, w, b)
y_test_pred = predict(x_test, w, b)

train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"Train MSE: {train_mse}")
print(f"Train MAE: {train_mae}")
print(f"Test MSE: {test_mse}")
print(f"Test MAE: {test_mae}")

def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    return r2

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train R^2 Score: {train_r2}")
print(f"Test R^2 Score: {test_r2}")