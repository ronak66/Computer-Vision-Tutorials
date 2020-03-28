import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import pickle

class BikeHorseClassifier:

    def __init__(self,x_train,y_train,num_of_clusters):
        self.num_of_clusters = num_of_clusters
        self.x_train = x_train
        self.y_train = y_train
        self.data_processing()
        self.create_features()

        return

    def show_image(self,img):
        cv2.imshow("output",img)
        cv2.imwrite('./static/whitening.png',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

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


    def data_processing(self):
        print("Processing Data","-"*(100-len("Processing Data")))
        sift_keypoints = []
        for image in self.x_train :
            # image = self.resize2SquareKeepingAspectRation(image,150)
            image = cv2.resize(image,(150,150))
            image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SURF_create()
            kp, descriptors = sift.detectAndCompute(image,None)
            sift_keypoints.append(descriptors)

        sift_keypoints = np.concatenate(sift_keypoints, axis=0)
        self.kmeans = KMeans(n_clusters = self.num_of_clusters).fit(sift_keypoints)
        print("Processing Data Complete","-"*(100-len("Processing Data Complete")))
        return self.kmeans

    def calculate_histogram(self,dataset,model,n):
        feature_vectors=[]

        for image in dataset :
            # image = image.astype('uint8')
            # image = self.resize2SquareKeepingAspectRation(image,150)
            image = cv2.resize(image,(150,150))
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #SIFT extraction
            sift = cv2.xfeatures2d.SURF_create()
            kp, descriptors = sift.detectAndCompute(image,None)
            #classification of all descriptors in the model
            predict_kmeans = model.predict(descriptors)
            #calculates the histogram
            hist, bin_edges = np.histogram(predict_kmeans, bins = n)
            #histogram is the feature vector
            feature_vectors.append(hist)

        feature_vectors=np.asarray(feature_vectors)

        return np.array(feature_vectors) 

    def create_features(self):
        print("Extracting Features","-"*(100-len("Extracting Features")))
        self.x_feat_train = self.calculate_histogram(self.x_train,self.kmeans,self.num_of_clusters)
        print("Extracting Features Completed","-"*(100-len("Extracting Features Completed")))
        return self.x_feat_train


    def train(self,classifier='RandomForest'):
        print("Training Model","-"*(100-len("Training Model")))
        if(classifier == 'RandomForest'):

            # self.rf =KNeighborsClassifier(n_neighbors=5, metric='euclidean')
            self.rf = RandomForestClassifier(n_estimators = 1000)
            cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
            scores = cross_val_score(self.rf, self.x_feat_train, self.y_train,cv=cv)
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

            xTrain, xTest, yTrain, yTest = train_test_split(self.x_feat_train, self.y_train, test_size = 0.2, random_state = 0)
            self.rf.fit(xTrain, yTrain)

        y_pred = self.rf.predict(xTest)
        accuracy = accuracy_score(y_pred, yTest)
        print("Acuuracy of Model: {}".format(accuracy))
        return confusion_matrix(yTest, y_pred)


    
    @staticmethod
    def load_dataset(path, classes,start=0, amount=500):
        print("Loading Data","-"*(100-len("Loading Data")))
        count=0
        class_images = []
        for cls in classes :
            cls_imgs = []
            amounts=amount
            img_names = os.listdir(path + cls + "/")
            for i in range(start,len(img_names)) :
                if(amounts==0):
                    break
                amounts-=1
                try:
                    # cls_imgs.append(cv2.resize(cv2.imread(path + cls + "/" + img_names[i]), (125, 125)))
                    cls_imgs.append(cv2.imread(path + cls + "/" + img_names[i]))
                except:
                    count+=1
            class_images.append(np.array(cls_imgs))
        print(str(count)+' images encountered error')
        print("Loading Data Complete","-"*(100-len("Loading Data Complete")))
        return class_images



class CIFAR10:

    def __init__(self,path):
        x_train, y_train = self.load_cfar10_batch(path,1)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        self.show_image(x_train[0])
        print("Number of Training Images: {}".format(x_train.shape[0]))
        classifier = BikeHorseClassifier(x_train, y_train, 10)
        print(classifier.train())
        return

    @staticmethod
    def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
        with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')
            
        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)

        labels = batch['labels']
        return features, labels

    def show_image(self,img):
        cv2.imshow("output",img)
        cv2.imwrite('./static/whitening.png',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

if __name__ == '__main__':
    
    data = BikeHorseClassifier.load_dataset("data/", ["Bikes", "Horses"])
    y_train = np.concatenate([np.ones(len(data[0])), np.zeros(len(data[1]))])
    x_train = np.concatenate(data)

    print("Number of Training Images: {}".format(x_train.shape[0]))
    classifier = BikeHorseClassifier(x_train, y_train, 10)
    print(classifier.train())
                              

    CIFAR10('data/cifar-10-batches-py')
