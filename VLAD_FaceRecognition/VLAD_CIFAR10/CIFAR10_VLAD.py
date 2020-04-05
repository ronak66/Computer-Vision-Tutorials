import cv2
import numpy as np
import pandas as pd
import pickle

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from VLAD import VLAD

class CIFAR10:
    
        def __init__(self,path,number_of_clusters):
            x_train, y_train = self.load_cfar10_batch(path,1)
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
            print("Number of Training Images: {}".format(x_train.shape[0]))
            vlad = VLAD(self.xTrain, self.xTest)
            (self.x_feat_train,self.x_feat_test) = vlad.vlad_features(number_of_clusters)
    
        def load_cfar10_batch(self,cifar10_dataset_folder_path, batch_id):
            for i in range(1,6):
                with open(cifar10_dataset_folder_path + '/data_batch_' + str(i), mode='rb') as file:
                    # note the encoding type is 'latin1'
                    batch = pickle.load(file, encoding='latin1')
                    
                feature = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
                label = batch['labels']
                
                feature = np.array(feature)
                label = np.array(label)
                if(i==1):    
                    features = feature
                    labels = label
                else:
                    features = np.vstack((features,feature))
                    labels = np.concatenate((labels,label))
            return features, labels
        
        def random_forest(self):
            print("Training Model","-"*(100-len("Training Model")))

            # self.rf =KNeighborsClassifier(n_neighbors=5, metric='euclidean')
            self.rf = RandomForestClassifier(n_estimators = 1000)
            self.rf.fit(self.x_feat_train, self.yTrain)

            y_pred = self.rf.predict(self.x_feat_test)
            accuracy = accuracy_score(y_pred, self.yTest)
            print("Acuuracy of Model: {}".format(accuracy))
            return confusion_matrix(self.yTest, y_pred)
        
        def SVM(self):
            self.clf = svm.LinearSVC(multi_class='ovr')
            self.clf.fit(self.x_feat_train, self.yTrain)

            y_pred = self.clf.predict(self.x_feat_test)
            accuracy = accuracy_score(y_pred, self.yTest)
            print("Acuuracy of Model: {}".format(accuracy))
            return confusion_matrix(self.yTest, y_pred)


if __name__ == '__main__':
    x = CIFAR10("../assignment2/data/cifar-10-batches-py",number_of_clusters=60)
    x.SVM()