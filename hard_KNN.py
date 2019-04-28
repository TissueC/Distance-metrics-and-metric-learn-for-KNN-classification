# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:46:10 2019

@author: Administrator
"""


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time
import os




def cosine(X,Y):
    return 1-np.sum(X*Y)/(np.sqrt(np.sum(X**2))*np.sqrt(np.sum(Y**2)))

data=np.load('Features/features.npy')
labels=np.load('Labels/labels.npy')
Models_Dir='Models'
Accuracy_Dir='Accuracy'
Figures_Dir='Figures'


distance_metric_list=['chebyshev','euclidean',
                      'manhattan',cosine ,
                      'hamming','canberra','braycurtis'][3:]

metric_name=['chebyshev','euclidean',
             'manhattan','cosine','hamming',
             'canberra','braycurtis'][3:]

n_neighbors_candidate=np.arange(3,21)

train_ratio=0.6

all_idx=np.arange(data.shape[0])
np.random.shuffle(all_idx)
train_idx=all_idx[:int(train_ratio*data.shape[0])]
test_idx=all_idx[int(train_ratio*data.shape[0]):][:1000]

is_plot=True
is_save_figure=True # invalid if is_plot==false and is_save_figure==true
is_save_accuracy=True
is_save_models=False 


#kf = KFold(n_splits=8
#for train_index, test_index in kf.split(data):

def main():
    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    start_time=time.time()
    idx=0
    for metric in distance_metric_list: # three part: [0,3],[3,6],[6,9]
        accuracy_list=list()
        for n_neighbors in n_neighbors_candidate:
            neigh=KNeighborsClassifier(n_neighbors=n_neighbors,
                                       metric=metric,n_jobs=-1)
            neigh.fit(X_train,y_train)
            if is_save_models:
                np.save(os.path.join(Models_Dir,
                                     metric_name[idx]+str(n_neighbors)+'.npy'),
                    neigh.get_params())
            
            pred_y=neigh.predict(X_test)
            print(pred_y,y_test)
            accuracy=np.bincount(pred_y==y_test)
            accuracy_list.append(accuracy[1]/np.sum(accuracy))
            print(metric_name[idx],n_neighbors,accuracy[1]/np.sum(accuracy))
        
        accuracy_list=np.array(accuracy_list)
        if is_save_accuracy:
            np.save(os.path.join(Accuracy_Dir, metric_name[idx]+'.npy'),
                    accuracy_list)
        print('accuracy: \n',accuracy_list)
        if is_plot:
            plt.plot(n_neighbors_candidate,accuracy_list)
            plt.title(metric_name[idx])
            plt.xlabel('numbor of neighbors')
            plt.ylabel('accuracy')
            if is_save_figure:
                plt.savefig(os.path.join(Figures_Dir, 
                                         metric_name[idx]+'.pdf'))
            plt.figure()
            
        idx+=1
        print('time: ',time.time()-start_time)

if __name__=='__main__':
    main()