"""feature size reduction by PCA"""

import numpy as np
import matplotlib.pyplot as plt

def covariance_matrix(X1, X2):
    mu1 = X1.mean()
    mu2 = X2.mean()
    print(mu1, mu2)
    A = 0
    B = 0
    C = 0
    M = np.zeros([2, 2])
    for i in range(len(X1)):
        A += (X1[i] - mu1) ** 2
        B += (X2[i] - mu2) ** 2
        C += (X1[i] - mu1) * (X2[i] - mu2)
        print(B)
    Amoy = A / len(X1)
    Bmoy = B / len(X1)
    Cmoy = C / len(X1)
    M[0][0] = Amoy
    M[1][1] = Bmoy
    M[0][1] = Cmoy
    M[1][0] = Cmoy
    print("covariance matrix:")
    print(M)
    return M

def f(t, mu1, mu2, k):
    return (1 / k) * (t - mu1) + mu2

def write_data(values, filename):
    """Write the n most frequent words to a file."""
    f = open(filename, "w")
    for value in values:
        print(value, file=f)
    f.close()


data = np.loadtxt("TrainingSet")
X1 = data[:, 3]
X2 = data[:, 7]
covariance_matrix(X1, X2)
mu1 = X1.mean()
mu2 = X2.mean()
k = 2027.4267096              # ((directive coefficient)^-1 of eigenvector (which has the highest eigenvalue):  2027.4267096 for PCA Fusion (3,7) Train -12.0028168811 for PCA Fusion (0,8) Train
t1 = np.arange(0, 25, 0.1)
plt.plot(t1, f(t1, mu1, mu2, k), 'k')
plt.scatter(X1, X2)
plt.show()
X1proj = np.zeros(len(X1))
X2proj = np.zeros(len(X1))
for i in range(len(X1)):
    Xsol = (X2[i] + k * X1[i] - mu2 + mu1 / k) / (k + 1 / k)
    X1proj[i] = Xsol
    X2proj[i] = f(Xsol, mu1, mu2, k)
t1 = np.arange(0, 25, 0.1)
plt.plot(t1, f(t1, mu1, mu2, k), 'k')
plt.scatter(X1proj, X2proj)
plt.show()
Xnew = np.sqrt(X1proj ** 2 + X2proj ** 2)
Ynew = np.zeros(len(X1))
plt.scatter(Xnew, Ynew)
plt.show()
write_data(Xnew, 'PCA features fusion/PCA Fusion (3,7) Train')


dataval = np.loadtxt("ValidationSet")
X1 = dataval[:, 3]
X2 = dataval[:, 7]
X1proj = np.zeros(len(X1))
X2proj = np.zeros(len(X1))
for i in range(len(X1)):
    Xsol = (X2[i] + k * X1[i] - mu2 + mu1 / k) / (k + 1 / k)
    X1proj[i] = Xsol
    X2proj[i] = f(Xsol, mu1, mu2, k)
t1 = np.arange(0, 25, 0.1)
plt.plot(t1, f(t1, mu1, mu2, k), 'k')
plt.scatter(X1proj, X2proj)
plt.show()
Xnew = np.sqrt(X1proj ** 2 + X2proj ** 2)
Ynew = np.zeros(len(X1))
plt.scatter(Xnew, Ynew)
plt.show()
write_data(Xnew, 'PCA features fusion/PCA Fusion (3,7) Validation')

datatest = np.loadtxt("TestSet")
X1 = datatest[:, 3]
X2 = datatest[:, 7]
X1proj = np.zeros(len(X1))
X2proj = np.zeros(len(X1))
for i in range(len(X1)):
    Xsol = (X2[i] + k * X1[i] - mu2 + mu1 / k) / (k + 1 / k)
    X1proj[i] = Xsol
    X2proj[i] = f(Xsol, mu1, mu2, k)
t1 = np.arange(0, 25, 0.1)
plt.plot(t1, f(t1, mu1, mu2, k), 'k')
plt.scatter(X1proj, X2proj)
plt.show()
Xnew = np.sqrt(X1proj ** 2 + X2proj ** 2)
Ynew = np.zeros(len(X1))
plt.scatter(Xnew, Ynew)
plt.show()
write_data(Xnew, 'PCA features fusion/PCA Fusion (3,7) Test')