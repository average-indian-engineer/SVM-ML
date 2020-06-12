
import h5py
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

rbfSigma = 0.1

def readFile(filename):
    with h5py.File(filename, 'r') as f:
        a_group_key = list(f.keys())
        X = list(f[a_group_key[0]])
        y = list(f[a_group_key[1]])
    return np.array(X), np.array(y)

def form_mesh(x, y, h=0.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    return np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

def linear(X1, X2):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.flatten()
            x2 = x2.flatten()
            gram_matrix[i, j] = np.dot(x1, x2)
    return gram_matrix

def rbfKernel(X1, X2):
    gamma = 1 / float( 2*(rbfSigma**2))
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.flatten()
            x2 = x2.flatten()
            gram_matrix[i, j] =  np.exp(- np.sum(np.power((x1 - x2), 2) ) * gamma )
    return gram_matrix

class SVM():

    def __init__(self, kernel = linear, C = 1, sigma=None):
        self.C = C
        self.kernel = kernel
        self.sigma = sigma
        self.clf = svm.SVC(kernel=self.kernel, C = self.C)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def accuracy(self, X, y):
        m = X.shape[0]
        return round(np.sum(self.predict(X) == y) / m, 3)

    def custom_predict(self, pX, X):
        m, n = pX.shape
        b = self.clf._intercept_[0]
        a = self.clf.dual_coef_[0]
        take_feature = self.clf.support_
        sv = X[take_feature]
        val = np.zeros(m)
        for j in range(m):
            for i in range(len(sv)):
                s = sv[i].reshape(n, 1)
                p = pX[j].reshape(n, 1)
                val[j] += a[i] * self.kernel(s.T, p.T)
        pred = (val - b >= 0)*1
        return pred
    
    def custom_accuracy(self, X, y, xTrain):
        pred = self.custom_predict(X, xTrain)
        return round(np.mean(pred == y), 3)

    def __str__(self):
        print('sv = ' + str(self.clf.support_vectors_))
        print('nv = ' + str(self.clf.n_support_))
        print('a = ' + str(self.clf.dual_coef_))
        print('a.shape = ' + str(self.clf.dual_coef_.shape))
        print('b = ' + str(self.clf._intercept_))
        print('cs = ' + str(self.clf.classes_))
        print(str(self.clf.support_))
        return ""

def get_data(fItr):
    filename = "./dataset_q1/data_" + str(fItr) + ".h5"
    X, y = readFile(filename)
    return X, y

def plot_graph(X, y, clfs, h, s, X_b, y_b):
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = form_mesh(X0, X1, h=h)
    meshVals = np.c_[xx.ravel(), yy.ravel()]
    Z = None
    try:
        Zs = []
        for model in clfs:
            Z = model.predict(meshVals)
            Zs.append(Z)
        Zs = np.array(Zs)
        Z = np.argmax(Zs, axis=0)
    except:
        Z = clfs.predict(meshVals)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=s, cmap=plt.cm.coolwarm)
    try:
        plt.scatter(X_b[:, 0], X_b[:, 1], c=y_b, s=20, cmap=plt.cm.cool)
    except:
        pass
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title("Decision Boundary Plot")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def get_model(X, y, ker, C = 1, sigma = 1, h = 0.1, visualize=True, s=None, X_b = [], y_b = []):
    global rbfSigma 
    m, n = X.shape
    rbfSigma = sigma
    clf = SVM(kernel=ker, C=C, sigma=sigma)
    clf.fit(X, y)
    
    if(visualize):
        plot_graph(X, y, clf, h, s, X_b, y_b)

    return clf

def getOutliers(X, y, f):
    scaler = StandardScaler()
    X0 = X[y == 0]
    y0 = y[y == 0]
    X1 = X[y == 1]
    y1 = y[y == 1]
    X0t = scaler.fit_transform(X0)
    X1t = scaler.fit_transform(X1)
    out0 = np.where(np.absolute(X0t) > f[0])[0]
    out1 = np.where(np.absolute(X1t) > f[1])[0]
    clusters0 = np.zeros(X0.shape[0])
    clusters1 = np.zeros(X1.shape[0])
    clusters0[out0] = -1
    clusters1[out1] = -1
    X0, y0, X_b0, y_b0 = refineData(X0, y0, clusters0)
    X1, y1, X_b1, y_b1 = refineData(X1, y1, clusters1)
    X = np.vstack((X0, X1))
    X_b = np.vstack((X_b0, X_b1))
    y = np.append(y0, y1)
    y_b = np.append(y_b0, y_b1)
    X, y = shuffle(X, y)
    return X, y, X_b, y_b

def refineData(X, y, f):
    X_b = X[f == -1]
    y_b = y[f == -1]
    X = np.delete(X, np.where(f == -1), axis = 0)
    y = np.delete(y, np.where(f == -1), axis = 0)
    return X, y, X_b, y_b

if __name__ == '__main__':
    
    X, y = get_data(1)
    clf1 = get_model(X, y, ker=rbfKernel, C=100, sigma=1, h=0.05, s=5)
    
    X, y = get_data(2)
    clf2 = get_model(X, y, ker=rbfKernel, C=1000, sigma=1, h=0.05, s=5)

    X, y = get_data(3)
    clf3 = []
    Y = y
    for i in range(3):
        y = np.copy(Y)
        y[y == (i + 2) % 3] = -1
        y[y == (i + 1) % 3] = -1
        y[y == i] = 1
        y[y == -1] = 0
        clf = get_model(X, y, ker=linear, visualize=False)
        clf3.append(clf)
    plot_graph(X, Y, clf3, h=0.3, s=10, X_b = [], y_b = [])

    X, y = get_data(4)
    X, y, X_b, y_b = getOutliers(X, y, (1.8, 1.9))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf4Linear = get_model(X_train, y_train, ker=linear, s=1, X_b=X_b, y_b=y_b)
    clf4Rbf = get_model(X_train, y_train, ker=rbfKernel, C=0.01, sigma=1, h=0.1, s=1, X_b=X_b, y_b=y_b)
    accuracyLinear_train = clf4Linear.custom_accuracy(X_train, y_train, X_train)
    accuracyLinear_test = clf4Linear.custom_accuracy(X_test, y_test, X_train)
    accuracyRbf_train = clf4Rbf.custom_accuracy(X_train, y_train, X_train)
    accuracyRbf_test = clf4Rbf.custom_accuracy(X_test, y_test, X_train)
    print("Accuracy for Prediciton with Custom Method")
    print(" "*10, "Linear".center(20), "RBF".center(20))
    print("Train".center(10), str(accuracyLinear_train).center(20), str(accuracyRbf_train).center(20))
    print("Test".center(10), str(accuracyLinear_test).center(20), str(accuracyRbf_test).center(20))
    accuracyLinear_train = clf4Linear.accuracy(X_train, y_train)
    accuracyLinear_test = clf4Linear.accuracy(X_test, y_test)
    accuracyRbf_train = clf4Rbf.accuracy(X_train, y_train)
    accuracyRbf_test = clf4Rbf.accuracy(X_test, y_test)
    print("Accuracy for Prediciton with Inbuilt Method")
    print(" "*10, "Linear".center(20), "RBF".center(20))
    print("Train".center(10), str(accuracyLinear_train).center(20), str(accuracyRbf_train).center(20))
    print("Test".center(10), str(accuracyLinear_test).center(20), str(accuracyRbf_test).center(20))
    
    X, y = get_data(5)
    X, y, X_b, y_b = getOutliers(X, y, (1.8, 1.9))    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf5Linear = get_model(X_train, y_train, ker=linear, s=1, X_b=X_b, y_b=y_b)
    clf5Rbf = get_model(X_train, y_train, ker=rbfKernel, C=10, sigma=2, h=0.1, s=1, X_b=X_b, y_b=y_b)
    accuracyLinear_train = clf5Linear.custom_accuracy(X_train, y_train, X_train)
    accuracyLinear_test = clf5Linear.custom_accuracy(X_test, y_test, X_train)
    accuracyRbf_train = clf5Rbf.custom_accuracy(X_train, y_train, X_train)
    accuracyRbf_test = clf5Rbf.custom_accuracy(X_test, y_test, X_train)
    print("Accuracy for Prediciton with Custom Method")
    print(" "*10, "Linear".center(20), "RBF".center(20))
    print("Train".center(10), str(accuracyLinear_train).center(20), str(accuracyRbf_train).center(20))
    print("Test".center(10), str(accuracyLinear_test).center(20), str(accuracyRbf_test).center(20))
    accuracyLinear_train = clf5Linear.accuracy(X_train, y_train)
    accuracyLinear_test = clf5Linear.accuracy(X_test, y_test)
    accuracyRbf_train = clf5Rbf.accuracy(X_train, y_train)
    accuracyRbf_test = clf5Rbf.accuracy(X_test, y_test)
    print("Accuracy for Prediciton with Inbuilt Method")
    print(" "*10, "Linear".center(20), "RBF".center(20))
    print("Train".center(10), str(accuracyLinear_train).center(20), str(accuracyRbf_train).center(20))
    print("Test".center(10), str(accuracyLinear_test).center(20), str(accuracyRbf_test).center(20))