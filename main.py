from pixelhop2 import Pixelhop2
from cross_entropy import Cross_Entropy
from lag import LAG
from llsr import LLSR as myLLSR
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
import joblib
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time
import os
from sklearn.metrics import confusion_matrix
import itertools

def subsetChoose(X, y, ratio):
    '''
    :param X: whole dataset
    :param y: label refer to dataset
    :param ratio: how many data samples we want to select from whole datasets, ratio is percentage
    :return: sub data samples
    '''
    size = int(len(X)*ratio)
    sizePerClass = size // 10
    l = []
    for i in range(9):
        temp = np.random.choice(np.where(y_train==i)[0], sizePerClass)
        l.extend(temp)
    temp = np.random.choice(np.where(y_train==9)[0], size-9*sizePerClass)
    l.extend(temp)
    np.random.shuffle(l)
    return X[l], y[l]

def maxPool(X,pool_param):
    '''
    :param X: dataset, the form is [N_samples, Height, Width, N_channels]
    :param pool_param: the form is {"pool_height": 2, "pool_width": 2, "stride": 2}
    :return:
    '''
    N, H, W, C = X.shape
    HH, WW, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
    H_out = (H-HH)//stride+1
    W_out = (W-WW)//stride+1
    out = np.zeros((N, H_out, W_out, C))
    for i in range(H_out):
        for j in range(W_out):
            x_mask = X[:,i*stride:i*stride+HH,j*stride:j*stride+WW,:]
            out[:,i,j,:] = np.max(x_mask,axis=(1,2))
    return out

def Shrink(X, shrinkArg):
    '''
    :param X: dataset, the form is [N_samples, Height, Width, N_channels]
    :param shrinkArg: the form is {'func': xxx, 'win': xxx, 'stride':xxx, 'pool_param':xxx}
    :return: the form is [N_samples, Height, Width, N_channels]
    '''
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    ch = X.shape[-1]
    X = view_as_windows(np.ascontiguousarray(X), (1,win,win,ch), (1,stride,stride,ch))
    return maxPool(X.reshape(X.shape[0], X.shape[1], X.shape[2], -1), shrinkArg["pool_param"])

def Concat(X, concatArg):
    '''
    :param X: dataset, the form is [N_samples, Height, Width, N_channels]
    :param concatArg: {'func': xxx}
    :return:
    '''
    return X

def trainModule1(X, y, ratio):
    '''
    :param X: datasets
    :param y: corresponding labels
    :param ratio: of which module2 and module3 we choose to use
    :return:
    '''
    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None, 'cw':False},
            {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw':True},
            {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw':True}
           ]
    shrinkArgs = [{'func':Shrink, 'win':5, 'stride': 1, "pool_param":{"pool_height": 2, "pool_width": 2, "stride": 2}},
                 {'func': Shrink, 'win':5, 'stride': 1, "pool_param":{"pool_height": 2, "pool_width": 2, "stride": 2}},
                 {'func': Shrink, 'win':5, 'stride': 1, "pool_param":{"pool_height": 1, "pool_width": 1, "stride": 1}}]
    concatArg = {'func':Concat}
    Xtrain, ytrain = subsetChoose(X, y, ratio) # randomly select 10k training images(1000per class)
    p2 = Pixelhop2(depth=3, TH1=0.001, TH2=0.0001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
    p2.fit(Xtrain)
    joblib.dump(p2, "saved_models/module1.pkl")
    return p2

def featureExtraction(model, X, y, r):
    '''
    :param model: cross entropy class object
    :param X: A hop unit, [N_samples, Height, Width, N_channels]
    :param y：label refer to X
    :param r: Number of selected feature(Ns) for a Hop unit, here is the percentage
    :return: The index of the top 50 less cross_entropy feature
    '''
    X = X.reshape(len(X), -1)
    feat_ce = np.zeros(X.shape[-1])
    for k in range(X.shape[-1]):
        feat_ce[k] = model.KMeans_Cross_Entropy(X[:, k].reshape(-1, 1), y)
    l = list(feat_ce)
    index = np.argsort(l)
    return l[:int(len(l)*r)]

def fitModule2_3(X, y, model, ratio=1):
    '''
    :param X: datasets
    :param y: corresponding labels
    :param model: module 1 to get three hop units
    :param ratio: of which module2 and module3 we choose to use
    :return: predicted labels and accuracy
    '''
    if ratio != 1:
        X, y = subsetChoose(X, y, ratio)
    ce = Cross_Entropy(num_class=10, num_bin=5)
    output = model.transform(X)
    print("Hop Units Found")
    indexs = []
    features = []
    for i in range(len(output)):
        temp = output[i]
        index = featureExtraction(ce, temp, y, 0.5)
        indexs.append(index)
        Xtrain = temp.reshape(len(X), -1)[:, index]
        lag = LAG(encode='distance', num_clusters=[5,5,5,5,5,5,5,5,5,5], alpha=10, learner=myLLSR(onehot=False))
        ytrain = y.copy()
        lag.fit(Xtrain, ytrain)
        np.save("saved_models/index_%s_%s.npy"%(str(ratio), str(i+1)), index)
        joblib.dump(lag, "saved_models/lag_%s_%s.pkl"%(str(ratio), str(i+1)))
        Xtrain_trans = lag.transform(Xtrain)
        features.append(Xtrain_trans)
    print("Features Found")
    f = features[0]
    for i in range(1,len(features)):
        f = np.hstack((f, features[i]))
    param_grid = {
        'bootstrap': [True],
        'min_samples_split': [8, 10, 12],
        #'n_estimators': [2, 5, 10, 20]
        'n_estimators': [10, 20, 30, 40]
    }
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=clf,param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(f, y.ravel())
    best_grid = grid_search.best_estimator_
    joblib.dump(best_grid, "saved_models/rf_%s.pkl"%str(ratio))
    print("RandomForest Found")

def calAcc(X, y, model, ratio):
    '''
    :param X: datasets
    :param y: corresponding labels
    :param model: module 1 to get three hop units
    :param ratio: of which module2 and module3 we choose to use
    :return: predicted labels and accuracy
    '''
    ce = Cross_Entropy(num_class=10, num_bin=5)
    output = model.transform(X)
    features = []
    for i in range(len(output)):
        temp = output[i]
        indexFile = "saved_models/index_%s_%s.npy"%(str(ratio), str(i+1))
        index = np.load(indexFile)
        Xtest = temp.reshape(len(X), -1)[:, index]
        lagFile = "saved_models/lag_%s_%s.pkl"%(str(ratio), str(i+1))
        lag = joblib.load(lagFile)
        Xtest_trans = lag.transform(Xtest)
        features.append(Xtest_trans)
    f = features[0]
    for i in range(1,len(features)):
        f = np.hstack((f, features[i]))
    gridFile = "saved_models/rf_%s.pkl"%str(ratio)
    best_grid = joblib.load(gridFile)
    y_pred = best_grid.predict(f)
    accuracy = accuracy_score(y.ravel(),y_pred)
    return y_pred, accuracy

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=45)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    #train time and accuracy
    start = time.time()
    if os.path.exists("saved_models/module1.pkl"):
        module1 = joblib.load("saved_models/module1.pkl")
    else:
        module1 = trainModule1(X_train, y_train, 0.2)
    fitModule2_3(X_train, y_train, module1, ratio=1)
    _, accuracy = calAcc(X_train, y_train, module1, ratio=1)
    end = time.time()
    print("The training time is:", end - start)
    print("The training accuracy is:", accuracy)

    #test accuracy
    start = time.time()
    y_pred, accuracy = calAcc(X_test, y_test, module1, ratio=1)
    print("The test accuracy is:", accuracy)
    end = time.time()
    print("The training time is:", end - start)
