import numpy as np
import matplotlib.pyplot as plt

"""The point of this program is to vizualize the eventual correlations between each feature 
with all the other features."""

S = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
data = np.loadtxt("TrainingSet")
X = data[:, :11]
Y = data[:, 11]
datatest = np.loadtxt("TestSet")
Xtest = datatest[:, :11]
Ytest = datatest[:, 11]

#for i in range(11):
#    print(S[i], " vs Y")
#    plt.scatter(Xtest[:, i], Ytest)
#    plt.show()

for i in range(11):
    for j in range(10 - i):
        print(S[i], " vs ", S[i + j + 1])
        plt.scatter(X[:, i], X[:, i + j + 1])
        plt.show()
