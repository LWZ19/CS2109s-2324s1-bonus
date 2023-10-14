import warnings
import numpy as np

# Q3
print("Q3")
y = np.array([0, 0, 1])
y_pred = np.array([[0.4, 0.4, 0.6],
                   [0.1, 0.6, 0.9]])
n, m = y_pred.shape

MSE = np.sum(np.power(y - y_pred, 2), axis=1) / (2 * m)
MAE = np.sum(np.abs(y - y_pred), axis=1) / (2 * m)
BCE = np.sum(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred), axis=1) / m
print("MSE:", MSE)
print("MAE:", MAE)
print("BCE:", BCE)

# Q4
print("\nQ4")
w_cat = [4.2, -0.01, -0.12]
w_horse = [-20, -0.08, 35]
w_elephant = [-1250, 0.82, 0.9]
w = np.array([w_cat, w_horse, w_elephant])

X = np.array([[4.2, 0.4], [720, 2.4], [2350, 5.5]])
X = np.pad(X, ((0, 0), (1, 0)), constant_values=1)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    P = 1 / (1 + np.exp(-(X @ w.T)))
print("P:", P)
classify = np.full(m, 'cat', dtype=object)
classify[P[1] > P[0]] = 'horse'
classify[P[2] > np.max(P[:2], axis=0)] = 'elephant'
print("Classification:", classify)

# Q5
print("\nQ5")
pred_prob = np.array([0.01, 0.02, 0.05, 0.07, 0.08, 0.09, 0.11, 0.12,
                      0.18,
                      0.28, 0.39,
                      0.62, 0.72,
                      0.79,
                      0.87, 0.92, 0.95, 0.96, 0.97, 0.98])
y_test = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                   1,
                   0, 0,
                   1, 1,
                   0,
                   1, 1, 1, 1, 1, 1])
p = 0.5
y_pred = pred_prob >= p
y_actual = y_test == 1
TP = (y_pred & y_actual).sum()
TN = (~y_pred & ~y_actual).sum()
FP = (y_pred & ~y_actual).sum()
FN = (~y_pred & y_actual).sum()

precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 / ((1 / precision) + (1 / recall))
print("Precision:", precision)
print("Recall:", recall)
print("F1:", F1)
