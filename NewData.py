"""Building the new feature vector after PCA"""


import numpy as np

X1 = np.loadtxt('PCA Fusion (0,8) Train').T
X2 = np.loadtxt('PCA Fusion (3,7) Train').T
h = np.loadtxt('TrainingSet')
X3 = h[:, 1]
X4 = h[:, 2]
X5 = h[:, 4]
X6 = h[:, 5]
#X7 = h[:, 6]
X8 = h[:, 9]
X9 = h[:, 10]
Y = h[:, 11]

f = open('PCA Data/NewTrainSet', "w")
for i in range(2300):
    print(X1[i], X2[i], X3[i], X4[i], X5[i], X6[i], X8[i], X9[i], Y[i], file=f)
f.close()



X1 = np.loadtxt('PCA Fusion (0,8) Validation').T
X2 = np.loadtxt('PCA Fusion (3,7) Validation').T
h = np.loadtxt('PCA Data/ValidationSet')
X3 = h[:, 1]
X4 = h[:, 2]
X5 = h[:, 4]
X6 = h[:, 5]
#X7 = h[:, 6]
X8 = h[:, 9]
X9 = h[:, 10]
Y = h[:, 11]

f = open('NewValidationSet', "w")
for i in range(200):
    print(X1[i], X2[i], X3[i], X4[i], X5[i], X6[i], X8[i], X9[i], Y[i], file=f)
f.close()



X1 = np.loadtxt('PCA Fusion (0,8) Test').T
X2 = np.loadtxt('PCA Fusion (3,7) Test').T
h = np.loadtxt('PCA Data/TestSet')
X3 = h[:, 1]
X4 = h[:, 2]
X5 = h[:, 4]
X6 = h[:, 5]
#X7 = h[:, 6]
X8 = h[:, 9]
X9 = h[:, 10]
Y = h[:, 11]

f = open('NewTestSet', "w")
for i in range(200):
    print(X1[i], X2[i], X3[i], X4[i], X5[i], X6[i], X8[i], X9[i], Y[i], file=f)
f.close()