## -*- coding: utf-8 -*-
#"""
#Created on Wed Apr 17 13:30:25 2019
#
#@author: Administrator
#"""
#
#
import numpy as np
from metric_learn import LMNN,NCA,LFDA,MLKR,ITML_Supervised,LSML_Supervised,\
SDML_Supervised,RCA_Supervised,MMC_Supervised
from sklearn.neighbors import KNeighborsClassifier
import time
import os

X=np.load('Features/features.npy')
Y=np.load('Labels/labels.npy')
Accuracy_Dir='Accuracy'

all_idx=np.arange(X.shape[0])
np.random.shuffle(all_idx)
train_idx=all_idx[:int(0.6*X.shape[0])]
test_idx=all_idx[int(0.6*X.shape[0]):]



train_X=X[train_idx]
test_X=X[test_idx]
train_Y=Y[train_idx]
test_Y=Y[test_idx]


neigh=KNeighborsClassifier(n_neighbors=5,
                           weights='distance',
                           metric='euclidean')

#Accuracy_list=list()
Accuracy_list=list(np.load('Accuracy/metric_learn_accuracy.npy'))
#print('No metric learn(baseline)')
starttime=time.time()
#neigh.fit(train_X,train_Y)
#pred_y=neigh.predict(test_X)
#accuracy=np.bincount(pred_y==test_Y)
#print('baseline:',accuracy[1]/np.sum(accuracy),'  time:',time.time()-starttime)
#Accuracy_list.append(accuracy[1]/np.sum(accuracy))
#np.save(os.path.join(Accuracy_Dir,'metric_learn_accuracy.npy'),
#                    np.array(Accuracy_list))
#
#print('NCA start to fit...')
#nca = NCA(max_iter=1000, learning_rate=0.01)
#nca.fit(train_X, train_Y)
#new_X=nca.transform(X)
#new_train_X=new_X[train_idx]
#new_test_X=new_X[test_idx]
#
#neigh.fit(new_train_X,train_Y)
#new_pred_y=neigh.predict(new_test_X)
#accuracy=np.bincount(new_pred_y==test_Y)
#print('NCA: ',accuracy[1]/np.sum(accuracy),'  time:',time.time()-starttime)
#Accuracy_list.append(accuracy[1]/np.sum(accuracy))
#np.save(os.path.join(Accuracy_Dir,'metric_learn_accuracy.npy'),
#                    np.array(Accuracy_list))


print('LMNN start to fit...')
lmnn = LMNN(k=5, learn_rate=1e-6)
lmnn.fit(train_X, train_Y)
new_X=lmnn.transform(X)
new_train_X=new_X[train_idx]
new_test_X=new_X[test_idx]

neigh.fit(new_train_X,train_Y)
new_pred_y=neigh.predict(new_test_X)
accuracy=np.bincount(new_pred_y==test_Y)
print('LMNN: ',accuracy[1]/np.sum(accuracy),'  time:',time.time()-starttime)
Accuracy_list.append(accuracy[1]/np.sum(accuracy))
np.save(os.path.join(Accuracy_Dir,'metric_learn_accuracy.npy'),
                    np.array(Accuracy_list))


print('LFDA start to fit...')
lfda = LFDA(k=5)
lfda.fit(train_X, train_Y)
new_X=lfda.transform(X)
new_train_X=new_X[train_idx]
new_test_X=new_X[test_idx]

neigh.fit(new_train_X,train_Y)
new_pred_y=neigh.predict(new_test_X)
accuracy=np.bincount(new_pred_y==test_Y)
print('LFDA: ',accuracy[1]/np.sum(accuracy),'  time:',time.time()-starttime)
Accuracy_list.append(accuracy[1]/np.sum(accuracy))
np.save(os.path.join(Accuracy_Dir,'metric_learn_accuracy.npy'),
                    np.array(Accuracy_list))


print('MLKR start to fit...')
mlkr = MLKR()
mlkr.fit(train_X, train_Y)
new_X=mlkr.transform(X)
new_train_X=new_X[train_idx]
new_test_X=new_X[test_idx]

neigh.fit(new_train_X,train_Y)
new_pred_y=neigh.predict(new_test_X)
accuracy=np.bincount(new_pred_y==test_Y)
print('MLKR: ',accuracy[1]/np.sum(accuracy),'  time:',time.time()-starttime)
Accuracy_list.append(accuracy[1]/np.sum(accuracy))
np.save(os.path.join(Accuracy_Dir,'metric_learn_accuracy.npy'),
                    np.array(Accuracy_list))


print('ITML start to fit...')
itml = ITML_Supervised()
itml.fit(train_X, train_Y)
new_X=itml.transform(X)
new_train_X=new_X[train_idx]
new_test_X=new_X[test_idx]

neigh.fit(new_train_X,train_Y)
new_pred_y=neigh.predict(new_test_X)
accuracy=np.bincount(new_pred_y==test_Y)
print('ITML: ',accuracy[1]/np.sum(accuracy),'  time:',time.time()-starttime)
Accuracy_list.append(accuracy[1]/np.sum(accuracy))
np.save(os.path.join(Accuracy_Dir,'metric_learn_accuracy.npy'),
                    np.array(Accuracy_list))


print('LSML start to fit...')
lsml = LSML_Supervised()
lsml.fit(train_X, train_Y)
new_X=lsml.transform(X)
new_train_X=new_X[train_idx]
new_test_X=new_X[test_idx]

neigh.fit(new_train_X,train_Y)
new_pred_y=neigh.predict(new_test_X)
accuracy=np.bincount(new_pred_y==test_Y)
print('LSML: ',accuracy[1]/np.sum(accuracy),'  time:',time.time()-starttime)
Accuracy_list.append(accuracy[1]/np.sum(accuracy))
np.save(os.path.join(Accuracy_Dir,'metric_learn_accuracy.npy'),
                    np.array(Accuracy_list))


print('SDML start to fit...')
sdml = SDML_Supervised()
sdml.fit(train_X, train_Y)
new_X=sdml.transform(X)
new_train_X=new_X[train_idx]
new_test_X=new_X[test_idx]

neigh.fit(new_train_X,train_Y)
new_pred_y=neigh.predict(new_test_X)
accuracy=np.bincount(new_pred_y==test_Y)
print('SDML: ',accuracy[1]/np.sum(accuracy),'  time:',time.time()-starttime)
Accuracy_list.append(accuracy[1]/np.sum(accuracy))
np.save(os.path.join(Accuracy_Dir,'metric_learn_accuracy.npy'),
                    np.array(Accuracy_list))


print('RCA start to fit...')
rca = RCA_Supervised()
rca.fit(train_X, train_Y)
new_X=rca.transform(X)
new_train_X=new_X[train_idx]
new_test_X=new_X[test_idx]

neigh.fit(new_train_X,train_Y)
new_pred_y=neigh.predict(new_test_X)
accuracy=np.bincount(new_pred_y==test_Y)
print('RCA: ',accuracy[1]/np.sum(accuracy),'  time:',time.time()-starttime)
Accuracy_list.append(accuracy[1]/np.sum(accuracy))
np.save(os.path.join(Accuracy_Dir,'metric_learn_accuracy.npy'),
                    np.array(Accuracy_list))


print('MMC start to fit...')
mmc = MMC_Supervised()
mmc.fit(train_X, train_Y)
new_X=mmc.transform(X)
new_train_X=new_X[train_idx]
new_test_X=new_X[test_idx]

neigh.fit(new_train_X,train_Y)
new_pred_y=neigh.predict(new_test_X)
accuracy=np.bincount(new_pred_y==test_Y)
print('MMC: ',accuracy[1]/np.sum(accuracy),'  time:',time.time()-starttime)
Accuracy_list.append(accuracy[1]/np.sum(accuracy))
np.save(os.path.join(Accuracy_Dir,'metric_learn_accuracy.npy'),
                    np.array(Accuracy_list))





#train_X=X[train_idx]
#test_X=X[test_idx]
#train_X=rca.fit_transform(train_X,train_Y)
#neigh=KNeighborsClassifier(n_neighbors=5,
#                           weights='distance',
#                           metric=cosine)
#neigh.fit(train_X,train_Y)
#pred_y=neigh.predict(test_X)
#accuracy=np.bincount(pred_y==test_Y)
#print(accuracy[1]/np.sum(accuracy))

