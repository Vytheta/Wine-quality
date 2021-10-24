import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logreg_inference(X, w, b):
    logits = (X @ w) + b
    #print('logits', logits)
    P = sigmoid(logits)
    return P

def sum(Y, p):
    d = 0
    for i in range(Y.size):
        d += abs(Y[i] - p[i])
    return d


def logreg_train(X, Y):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    lr = 0.0003
    for step in range(500000):
        P = logreg_inference(X, w, b)
        grad_w = (X.T @ (P - Y)) / m
        grad_b = (P - Y).mean()
        w = w - lr * grad_w
        b = b - lr * grad_b
        #print(w, b)
        p = logreg_inference(X, w, b)
        n = 1 - sum(Y, p) / Y.size
        print(n)
    return w, b

data = np.loadtxt("TrainingSet")
X = data[:, :-1]
#print(X)
Y = data[:, -1]
w, b = logreg_train(X, Y)