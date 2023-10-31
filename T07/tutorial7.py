import numpy as np
from functools import reduce

def relu(x):
    return x * (x > 0)

class FconLayer(object):
    def __init__(self, W, a_fn=relu, has_bias=True):
        self.W = W
        self.a_fn = a_fn
        self.has_bias = has_bias

    def __call__(self, X_T):
        m, n = X_T.shape
        X_T_copy = np.copy(X_T)
        if self.has_bias:
            bias = np.ones((1, n))
            X_T_copy = np.vstack((bias, X_T_copy))
        return self.a_fn(np.transpose(self.W) @ X_T_copy)



l1 = FconLayer(np.array([
    [0.1, 0.1],
    [-0.1, 0.2],
    [0.3, -0.4]
]))
l2 = FconLayer(np.array([
    [0.1, 0.1],
    [0.5, -0.6],
    [0.7, -0.8]
]))

class NNetwork(object):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, X):
        return np.transpose(reduce(lambda x, y: y(x), self.layers, np.transpose(X)))

nn_q3 = NNetwork([l1, l2])
x_q3 = np.array([[2,3]])
y_q3 = np.array([[0.1, 0.9]])
y_pred = nn_q3(x_q3)

mse = ((y_pred - y_q3)**2).mean()
print(f"Q3 MSE: {mse}")

Ws = [
    np.array([
        [1.4, 0.6],
        [0.8, 0.6]
    ]),
    np.array([
        [2.1, -0.5],
        [0.7, 1.9]
    ]),
    np.array([
        [1.2, -2.2],
        [1.2, 1.3]
    ])
]
    
nn_q4_noact = NNetwork([ FconLayer(W, lambda x:x, has_bias=False) for i, W in enumerate(Ws) ])
nn_q4_silu = NNetwork([ FconLayer(W, (lambda x:x/(1+np.exp(-x))) if i < 2 else (lambda x:x), has_bias=False) for i, W in enumerate(Ws) ])
x_q4 = np.array([[1,0], [2,0]])

M = reduce(lambda x, y: np.transpose(y) @ np.transpose(x), Ws)
nn_q4_l1 = NNetwork([ FconLayer(M, lambda x:x, has_bias=False) ])

is_equal = np.allclose(nn_q4_noact(x_q4), nn_q4_l1(x_q4))
print("Q4 compressed M results is equal - {is_equal}")
print("Q4 counter example:")
print(nn_q4_silu(x_q4))

# Expected output:
# Q3 MSE: 0.48500000000000004
# Q4 compressed M results is equal - {is_equal}
# Q4 counter example:
# [[  3.05710826  -5.27270693]
#  [  7.72572945 -13.2458166 ]]
