from idlelib import history

import numpy as np
from ann_visualizer.visualize import ann_viz
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from sklearn.metrics import confusion_matrix
from keras.optimizers import SGD
import pandas as pd
from sklearn.utils import shuffle
from sklearn import svm

confut=[]
confution=[]
def perf_measure(y_actual,y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    count=0
    count1=0
    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            tp += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            fp += 1
        if y_actual[i] == y_pred[i] == 0:
            tn += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            fn += 1
        if y_actual[i]  == 1:
            count+=1
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen=tp/(tp+fn)
    spe=tn/(tn+fp)
    confut.append([tp,tn,fp,fn])
    count1=len(y_actual)-count
    confution.append([tp,tn,fp,fn,count,count1])
    return [np.around(acc*100,2),np.around(sen*100,2),np.around(spe*100,2)]

def ann(dataset,i):
    x = dataset.iloc[:, 0:48].values
    y = dataset.iloc[:, 48+i].values

    kf = KFold(n_splits=10, random_state=123, shuffle=True)
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=None)

    classifier = Sequential()
    node = 32
    classifier.add(Dense(48,activation='relu', input_dim=48))
    for j in range(3):
        classifier.add(Dense(node,activation='relu'))
        node = round(node / 2)

    classifier.add(Dense(output_dim=1, activation='sigmoid'))
    opt = SGD(lr=0.01, momentum=0.9)
    classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    history=classifier.fit(X_train, y_train, nb_epoch=75, batch_size=5, verbose=1,validation_data=(X_test, y_test))
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    conM.append((cm.flatten()).tolist())
    #ann_viz(classifier, title="My first neural network")
    print(classifier.summary())
    #print(cm)
    #print(perf_measure(y_test, y_pred))

    _, train_acc = classifier.evaluate(X_train, y_train, verbose=0)
    _, test_acc = classifier.evaluate(X_test, y_test, verbose=0)
    titel = ["CVD", "STR", "DR", "FCD"]
    pyplot.subplot(211)
    pyplot.title(titel[i]+'- Cross-Entropy Loss ', pad=-30)
    pyplot.plot(history.history['loss'], label=' Train')
    pyplot.plot(history.history['val_loss'], label=' Test ')
    pyplot.legend()
    # plot accuracy learning curves
    pyplot.subplot(212)
    pyplot.title(titel[i]+'- Accuracy', pad=-50)
    pyplot.plot(history.history['acc'], label=' Train')
    pyplot.plot(history.history['val_acc'], label=' Test ')
    pyplot.legend()
    #pyplot.show()
    pyplot.savefig("Result/"+titel[i]+".png")
    pyplot.clf()
    #print('Train: %.3f\nTest: %.3f' % (train_acc, test_acc))
    return perf_measure(y_test, y_pred)

dataFile=["Data/cvd_data.csv","Data/str_data.csv","Data/dr_data.csv","Data/fcd_data.csv"]
result=[]
conM=[]
for i in range(0,4):
    dataset = pd.read_csv(dataFile[i])
    result.append(ann(dataset,i))
print(result)
print(conM)
np.savetxt("Data/result.csv", result, delimiter=',')
np.savetxt("Data/confutation_matrix.csv", conM, delimiter=',')


