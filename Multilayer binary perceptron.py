import pvml
import matplotlib.pyplot as plt

Xtrain, Ytrain = pvml.load_dataset("TrainingSet")
Xval, Yval = pvml.load_dataset("ValidationSet")
Xtest, Ytest = pvml.load_dataset("TestSet")

net = pvml.MLP([11, 4, 2])

plt.ion()

train_accs = []
val_accs = []
for epoc in range(1000):
    net.train(Xtrain, Ytrain, lr=1e-7, lambda_=1e-7, momentum=0.99, steps=3000, batch=50)
    predictionsTrain = net.inference(Xtrain)[0]
    train_acc = (predictionsTrain == Ytrain).astype(int)
    train_acc = train_acc.mean()
    predictionsVal = net.inference(Xval)[0]
    print(predictionsTrain)
    val_acc = (predictionsVal == Yval).mean()
    print(train_acc * 100, val_acc * 100)
    train_accs.append(train_acc * 100)
    val_accs.append(val_acc * 100)
    plt.clf()
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.legend(["train", "val"])
    plt.xlabel("epocs")
    plt.ylabel("accuracy (%)")
    plt.pause(0.05)

plt.ioff()
plt.show()