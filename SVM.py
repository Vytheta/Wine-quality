import numpy as np

def svm_inference(X, w, b):
    logits = X @ w + b
    labels = (logits > 0).astype(int)
    return labels, logits

def hinge_loss(labels, logits):
    loss = np.maximum(0, 1 - (2 * labels - 1) * logits)
    return loss.mean()

def svm_train(X, Y, lambda_, lr=0.0001, steps=500000, init_w=None, init_b=0):
    m, n = X.shape
    w = (init_w if init_w is not None else np.zeros(n))
    b = init_b
    C = (2 * Y) - 1
    for step in range(steps):
        labels, logits = svm_inference(X, w, b)
        hinge_diff = -C * ((C * logits) < 1)
        grad_w = (hinge_diff @ X) / m + lambda_ * w
        grad_b = hinge_diff.mean()
        w -= lr * grad_w
        b -= lr * grad_b
        #print(w, b)
        hingeloss = np.maximum(1 - C * logits, 0)
        J = hingeloss.mean() + ((w.T) @ w) * lambda_ / 2
        print(J)
        #print(hinge_loss(labels, logits))
    return w, b


data = np.loadtxt("TrainingSet")
X = data[:, :-1]
Y = data[:, -1]
lambda_ = 0.00000001
beta = svm_train(X, Y, lambda_)
inf = svm_inference(X, beta[0], beta[1])
accuracy = (Y == inf[0]).mean()
print("train accuracy", accuracy * 100, "%")
validation_data = np.loadtxt("ValidationSet")
Xv = validation_data[:, :-1]
Yv = validation_data[:, -1]
inf = svm_inference(Xv, beta[0], beta[1])
accuracy = (Yv == inf[0]).mean()
print("validation accuracy", accuracy * 100, "%")