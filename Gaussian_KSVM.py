import numpy as np

def ksvm_Gausstrain(X, Y, d, lambda_, lr=0.1, steps=30000, init_b=0):
    K = GaussKernel(X, X, d)
    m, n = X.shape
    alpha = np.zeros(m)
    b = (init_b if init_b is not None else 0)
    C = (2 * Y) - 1
    for step in range(steps):
        ka = K @ alpha
        logits = ka + b
        hinge_diff = -C * ((C * logits) < 1)
        grad_alpha = (hinge_diff @ K) / m + lambda_ * ka
        grad_b = hinge_diff.mean()
        alpha -= lr * grad_alpha
        b -= lr * grad_b
        hinge_loss = np.maximum(1 - C * logits, 0)
        J = hinge_loss.mean() + (alpha @ (K @ alpha)) * lambda_ / 2
        print(J)
        #q = ksvm_Gaussinference(X, X, alpha, b, d)[0]
        #S = (Y == q).astype(int)
        #accuracy = S.mean()
        #print(accuracy)
    return alpha, b


def ksvm_Gaussinference(X, Xtrain, alpha, b, d):
    K  = GaussKernel(X, Xtrain, d)
    logits = K @ alpha + b
    labels = (logits > 0).astype(int)
    return labels, logits


def GaussKernel(X1, X2, d):
    qX1 = (X1 ** 2).sum(1, keepdims=True)
    qX2 = (X2 ** 2).sum(1, keepdims=True)
    return np.exp(-d * (qX1 - 2 * X1 @ X2.T + qX2.T))

def cross_validation(k, X, Y, d, lambda_):               # I tried once a k cross validation, but the results dit not really improve
    m = X.shape[0]
    folds = np.arange(m) % k
    np.random.shuffle(folds)
    real_predictions = 0
    for fold in range(k):
        Xtrain = X[folds != fold, :]
        Ytrain = Y[folds != fold]
        params = ksvm_Gausstrain(Xtrain, Ytrain, d, lambda_)
        Xval = X[folds == fold, :]
        Yval = Y[folds == fold]
        predictions = ksvm_Gaussinference(Xval, Xtrain, params[0], params[1], d)[0]
        real_predictions += (predictions == Yval).sum()
    preds = real_predictions / m
    print("validation accuracy", preds * 100, "%")
    return real_predictions / m



data = np.loadtxt("PCA Data/NewTrainSet")
X = data[:, :-1]
Y = data[:, -1]
d = 0.1
lambda_ = 0.0001
beta = ksvm_Gausstrain(X, Y, d, lambda_)
inf = ksvm_Gaussinference(X, X, beta[0], beta[1], d)
accuracy = (Y == inf[0]).mean()
print("train accuracy", accuracy * 100, "%")
validation_data = np.loadtxt("PCA Data/NewValidationSet")
Xv = validation_data[:, :-1]
Yv = validation_data[:, -1]
inf = ksvm_Gaussinference(Xv, X, beta[0], beta[1], d)
accuracy = (Yv == inf[0]).mean()
print("validation accuracy", accuracy * 100, "%")
test_data = np.loadtxt("PCA Data/NewTestSet")
Xt = test_data[:, :-1]
Yt = test_data[:, -1]
inf = ksvm_Gaussinference(Xt, X, beta[0], beta[1], d)
accuracy = (Yt == inf[0]).mean()
print("test accuracy", accuracy * 100, "%")

#data1 = np.loadtxt('tottrain')                 # tottrain combines original train data and validation
#X = data1[:, :-1]
#Y = data1[:, -1]
#d = 0.3
#lambda_ = 0.000001
#preds = cross_validation(5, X, Y, d, lambda_)

