import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

def kmeans(x_train,num_of_clusters=10,num_of_kp=40):
    sift_keypoints = []
    for image in x_train :
        image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(image,None)
        sift_keypoints.append(descriptors)


    sift_keypoints = np.concatenate(sift_keypoints, axis=0)
    kmeans = KMeans(n_clusters = num_of_clusters).fit(sift_keypoints)
    print(sift_keypoints.shape)
    return kmeans

def sortresponse1(kp1):
    return kp1[0].response

def calculate_histogram(images, model,n,num_of_kp):

    feature_vectors=[]
    i=0
    for image in images :
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #SIFT extraction
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(image,None)
        predict_kmeans = model.predict(descriptors)
        #calculates the histogram
        hist, bin_edges = np.histogram(predict_kmeans, bins = n)
        hist = list(hist)
        hist.append(np.mean(image))
        feature_vectors.append(hist)
        i+=1
    feature_vectors=np.asarray(feature_vectors)

    return np.array(feature_vectors)


def load_dataset(path, classes,start=0, amount=200):
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
                cls_imgs.append(cv2.resize(cv2.imread(path + cls + "/" + img_names[i]), (150, 150)))
            except Exception as e:
                count+=1
        class_images.append(np.array(cls_imgs))
    print(str(count)+' images encountered error')
    return class_images

def show_image(img):
    cv2.imshow("output",img)
    cv2.imwrite('./static/whitening.png',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    
    data = load_dataset("q2_dataset/", ["night", "faces", "landscapes"])
    y_train = np.concatenate([np.zeros(len(data[0])), np.ones(len(data[1])), np.ones(len(data[2]))+1])
    x_train = np.concatenate(data)
    xTrain, xTest, yTrain, yTest = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

    kmeans = kmeans(xTrain)
    x_feat_train = calculate_histogram(xTrain, kmeans,10,40)
    x_feat_test = calculate_histogram(xTest, kmeans,10,40)

    # rf = KNeighborsClassifier(n_neighbors=n, metric='euclidean')
    rf = RandomForestClassifier(n_estimators = 1000)
    rf.fit(x_feat_train, yTrain)
    y_pred = rf.predict(x_feat_test)
    print(accuracy_score(y_pred, yTest))
    print(multilabel_confusion_matrix(yTest,y_pred))
    cm = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(len(yTest)):
        cm[int(yTest[i])][int(y_pred[i])]+=1
    print(cm)
