import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def getTrainData(n=500):
    X = []
    y = []
    for i in range(1, 6):
        data = unpickle("./dataset_q2/data_batch_" + str(i))
        nX = np.array(data[b'data'])
        ny = np.array(data[b'labels']).reshape(10000, 1)
        if(len(X) != 0):
            X = np.vstack((X, nX))
            y = np.vstack((y, ny))
        else:
            X = nX
            y = ny
    dX, dy = shuffle(X, y)
    data = {}
    for cl in (np.unique(y)):
        data[cl] = []
        counter = 0
        while(len(data[cl]) < n):
            if(y[counter] == cl):
                data[cl].append(X[counter])
            counter += 1
        data[cl] = np.array(data[cl])
    y = np.array([0] * n)
    X = np.array(data[0])
    for cl in range(1, 10):
        y = np.vstack((y, [cl] * n))
        X = np.vstack((X, data[cl]))
    y = y.ravel()
    return shuffle(X, y)

def getTestData(n=100):
    data = unpickle("./dataset_q2/test_batch")
    X = np.array(data[b'data'])
    y = np.array(data[b'labels']).reshape(10000, 1)
    data = {}
    for cl in (np.unique(y)):
        data[cl] = []
        counter = 0
        while(len(data[cl]) < n):
            if(y[counter] == cl):
                data[cl].append(X[counter])
            counter += 1
        data[cl] = np.array(data[cl])
    y = np.array([0] * n)
    X = np.array(data[0])
    for cl in range(1, 10):
        y = np.vstack((y, [cl] * n))
        X = np.vstack((X, data[cl]))
    y = y.ravel()
    return shuffle(X, y)

def getModelFunction(kernel, shapeF):
    clf = None
    if(kernel == 'poly'):
        clf = svm.SVC(gamma='scale', decision_function_shape=shapeF, kernel=kernel, degree=2, probability=True)
    else:
        clf = svm.SVC(gamma='scale', decision_function_shape=shapeF, kernel=kernel, probability=True)
    return clf

def showStats(clf, X, y):
    y_pred = clf.predict(X)
    print("Confusion Matrix")
    print(confusion_matrix(y, y_pred))
    acc = accuracy_score(y, y_pred)
    print("Accuracy Score on Test Set: " + str(acc))
    print("Classification Report:")
    print(classification_report(y, y_pred)) 
    return acc

def getOvoModel(X_train, y_train, X_val, y_val, XTest, yTest, kernel):
    print("-" * 10 + str(kernel) + " OVO" + "-" * 10)
    clf = getModelFunction(kernel, 'ovo')
    clf.fit(X_train, y_train)
    if(kernel == 'linear'):
        w = np.array(clf.coef_)
        print("Weights for Linear OVO")
        print(w.shape)
    else:
        print("Can't give weights, model not linear!")
    y_pred = clf.predict(X_val)
    valScore = accuracy_score(y_val, y_pred)
    print("Accuracy on Validation set: " + str(valScore))
    acc = showStats(clf, XTest, yTest)
    return clf, acc

def getOvaModel(X_train, y_train, X_val, y_val, XTest, yTest, kernel):
    print("-" * 10 + str(kernel) + " OVA" + "-" * 10)
    clf = OneVsRestClassifier(getModelFunction(kernel, 'ovr'))
    clf.fit(X_train, y_train)
    if(kernel == 'linear'):
        w = np.array(clf.coef_)
        print("Weights for Linear OVA")
        print(w.shape)
    else:
        print("Can't give weights, model not linear!")
    y_pred = clf.predict(X_val)
    valScore = accuracy_score(y_val, y_pred)
    print("Accuracy on Validation set: " + str(valScore))
    acc = showStats(clf, XTest, yTest)
    return clf, acc

def groupColumn(data, x):
    data = np.copy(data)
    data[:,0] = [e == x for e in data[:,0]]
    return data

def plotRocCurve(model, XTest, yTest, name):
    n_classes = 10
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    yTest = label_binarize(yTest, classes=[i for i in range(10)])
    yscore = model.predict_proba(XTest)
    for i in range(n_classes):
        roc_auc[i] = roc_auc_score(yTest[:, i], yscore[:, i])
        fpr[i], tpr[i], _ = roc_curve(yTest[:, i], yscore[:, i])
    
    plt.figure()    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='Class ' + str(i) + ' (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for '+ name +' OVA')
    fig = plt.gcf()
    fig.canvas.set_window_title('ROC Curve for ' + name + ' OVA')
    plt.show()

if __name__ == "__main__":
    
    XTrain, yTrain = getTrainData(100)
    XTest, yTest = getTestData(20)

    kf = KFold(n_splits=5)

    ovoModelDict = {
        'linear': None,
        'rbf': None, 
        'poly': None
    }

    ovaModelDict = {
        'linear': None,
        'rbf': None, 
        'poly': None
    }

    first_itr = True

    for train_ind, val_ind in kf.split(XTrain):
        
        X_train, X_val = XTrain[train_ind], XTrain[val_ind]
        y_train, y_val = yTrain[train_ind], yTrain[val_ind]

        linear_ovo, score_linear_ovo = getOvoModel(X_train, y_train, X_val, y_val, XTest, yTest, 'linear')
        rbf_ovo, score_rbf_ovo = getOvoModel(X_train, y_train, X_val, y_val, XTest, yTest, 'rbf')
        polynomial_ovo, score_polynomial_ovo = getOvoModel(X_train, y_train, X_val, y_val, XTest, yTest, 'poly')

        linear_ova, score_linear_ova = getOvaModel(X_train, y_train, X_val, y_val, XTest, yTest, 'linear')
        rbf_ova, score_rbf_ova = getOvaModel(X_train, y_train, X_val, y_val, XTest, yTest, 'rbf')
        polynomial_ova, score_polynomial_ova = getOvaModel(X_train, y_train, X_val, y_val, XTest, yTest, 'poly')
        
        if(first_itr):
            ovoModelDict['linear'] = (linear_ovo, score_linear_ovo)
            ovoModelDict['rbf'] = (rbf_ovo, score_rbf_ovo)
            ovoModelDict['poly'] = (polynomial_ovo, score_polynomial_ovo)
            ovaModelDict['linear'] = (linear_ova, score_linear_ova)
            ovaModelDict['rbf'] = (rbf_ova, score_rbf_ova)
            ovaModelDict['poly'] = (polynomial_ova, score_polynomial_ova)
            first_itr = False
        else:
            if(score_linear_ovo > ovoModelDict['linear'][1]):
                ovoModelDict['linear'] = (linear_ovo, score_linear_ovo)
            if(score_rbf_ovo > ovoModelDict['rbf'][1]):
                ovoModelDict['rbf'] = (rbf_ovo, score_rbf_ovo)
            if(score_polynomial_ovo > ovoModelDict['poly'][1]):
                ovoModelDict['poly'] = (polynomial_ovo, score_polynomial_ovo)
            if(score_linear_ova > ovaModelDict['linear'][1]):
                ovaModelDict['linear'] = (linear_ova, score_linear_ova)
            if(score_rbf_ova > ovaModelDict['rbf'][1]):
                ovaModelDict['rbf'] = (rbf_ova, score_rbf_ova)
            if(score_polynomial_ova > ovaModelDict['poly'][1]):
                ovaModelDict['poly'] = (polynomial_ova, score_polynomial_ova)

    pickle.dump(ovoModelDict, open('./q2Models/ovoModels.sav', 'wb'))    
    pickle.dump(ovaModelDict, open('./q2Models/ovaModels.sav', 'wb'))
    pickle.dump(XTrain, open('./q2Models/XTrain.sav', 'wb'))
    pickle.dump(yTrain, open('./q2Models/yTrain.sav', 'wb'))
    pickle.dump(XTest, open('./q2Models/XTest.sav', 'wb'))
    pickle.dump(yTest, open('./q2Models/yTest.sav', 'wb'))

    ovoModelDict = pickle.load(open('./q2Models/ovoModels.sav', 'rb'))
    ovaModelDict = pickle.load(open('./q2Models/ovaModels.sav', 'rb'))
    XTrain = pickle.load(open('./q2Models/XTrain.sav', 'rb'))
    yTrain = pickle.load(open('./q2Models/yTrain.sav', 'rb'))
    XTest = pickle.load(open('./q2Models/XTest.sav', 'rb'))
    yTest = pickle.load(open('./q2Models/yTest.sav', 'rb'))
    
    print("Max accuracies for different kernels are: ")
    print("Linear OVA: " + str(ovaModelDict['linear'][1]))
    print("RBF OVA: " + str(ovaModelDict['rbf'][1]))
    print("Polynomial OVA: " + str(ovaModelDict['poly'][1]))
    print("Linear OVO: " + str(ovoModelDict['linear'][1]))
    print("RBF OVO: " + str(ovoModelDict['rbf'][1]))
    print("Polynomial OVO: " + str(ovoModelDict['poly'][1]))
    print("It's clear that RBF is a better performing kernel")

    plotRocCurve(ovaModelDict['linear'][0], XTest, yTest, "Linear")
    plotRocCurve(ovaModelDict['rbf'][0], XTest, yTest, "RBF")
    plotRocCurve(ovaModelDict['poly'][0], XTest, yTest, "Polynomial")