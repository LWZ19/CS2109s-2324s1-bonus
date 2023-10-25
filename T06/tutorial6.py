import numpy as np

# Code adapted from https://www.youtube.com/watch?v=T9UcK-TxQGw.

lr = 0.01
lambda_param = 0.01
n_iters = 5000

def fit(X, y):
    n_samples, n_features = X.shape
    bias, weight = 0, np.random.random(n_features)

    for _ in range(n_iters):
        for idx, x_i in enumerate(X):
            condition = y[idx] * (np.dot(x_i, weight) - bias) >= 1
            if condition:
                weight -= lr * (2 * lambda_param * weight)
            else:
                weight -= lr * (2 * lambda_param * weight - np.dot(x_i, y[idx]))
                bias -= lr * y[idx]

    return bias, weight


X = np.array([[-2, -2], [-2, 0], [0, 2], [1, 1], [3, 0]])
y = np.array([-1, -1, 1, 1, 1])

bias, weight = fit(X, y)
print("bias = ", bias)
print("weight = ", weight)
