import cv2
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans


class VLAD:

    def __init__(self,x_train,x_test):
        self.x_train = x_train
        self.x_test = x_test


    def cluster_formation(self,number_of_clusters):
        print("Performing KMeans Clustering","-"*(100-len("Performing KMeans Clustering")))
        keypoints = []
        counter=1
        bar = progressBar()
        for image in self.x_train :
            # image = self.resize2SquareKeepingAspectRation(image,150)
            bar.update(counter/len(list(self.x_train)))
            image = cv2.resize(image,(150,150))
            image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            surf = cv2.xfeatures2d.SURF_create()
            kp, descriptors = surf.detectAndCompute(image,None)
            keypoints.append(descriptors)
            counter+=1
        keypoints = np.concatenate(keypoints, axis=0)
        self.kmeans = KMeans(n_clusters = number_of_clusters).fit(keypoints)
        print("KMeans Clustering finished","-"*(100-len("KMeans Clustering finished")))
        return self.kmeans
    
    def vlad(self,descriptors,number_of_clusters):
        dist_sum = []
        for i in range(number_of_clusters):
            dist_sum.append(np.zeros(64))
        predict_kmeans = self.kmeans.predict(descriptors)
        cluster_centers = self.kmeans.cluster_centers_
        for i in range(len(descriptors)):
            dist_sum[predict_kmeans[i]] += (np.asarray(descriptors[i]) - cluster_centers[predict_kmeans[i]])
        dist_sum = np.asarray(dist_sum)
        V = dist_sum.flatten()
        V = V/np.sqrt(np.dot(V,V))
        return V

    
    def vlad_features(self,number_of_clusters):
        self.cluster_formation(number_of_clusters)
        print("Performing VLAD extraction","-"*(100-len("Performing VLAD extraction")))
        x_feat_train = []
        x_feat_test = []
        counter=1
        bar = progressBar()
        for n in range(2):
            if(n==0): dataset = self.x_train
            else: dataset = self.x_test
            for image in dataset:
                # image = self.resize2SquareKeepingAspectRation(image,150)
                bar.update(counter/(len(list(self.x_train))+len(list(self.x_test))))
                image = cv2.resize(image,(150,150))
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                surf = cv2.xfeatures2d.SURF_create()
                kp, descriptors = surf.detectAndCompute(image,None)
#                 V = VLADL(descriptors,self.kmeans)
                V = self.vlad(descriptors,number_of_clusters)
                if(n==0): x_feat_train.append(V)
                else: x_feat_test.append(V)
                counter+=1
        print("VLAD extraction finished","-"*(100-len("VLAD extraction finished")))
        return (x_feat_train,x_feat_test)

    def resize2SquareKeepingAspectRation(self,img, size, interpolation = cv2.INTER_AREA):
        h, w = img.shape[:2]
        c = None if len(img.shape) < 3 else img.shape[2]
        if h == w: return cv2.resize(img, (size, size), interpolation)
        if h > w: dif = h
        else: dif = w
        x_pos = int((dif - w)/2.)
        y_pos = int((dif - h)/2.)
        if c is None:
            mask = np.zeros((dif, dif), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
        return cv2.resize(mask, (size, size), interpolation)    


import progressbar
class progressBar:
    def __init__(self):
            self.bar = progressbar.ProgressBar(maxval=1, \
                                  widgets=[progressbar.Bar('=','[',']'), ' ', progressbar.Percentage()])
            self.bar.start()
    
    def update(self,value):
        self.bar.update(value)

    def restart(self):
        self.bar.start()