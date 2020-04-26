
#  Importing Libraries

import matlib.pyplot as plt
import cv2
import numpy as np
from numpy import save,load
import pandas as pd
import os,glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importing the Pre-processed Data Set

X = load('/home/ram/IIITB/Sem6/VR/Project1/Ajay/VR-Project-1/dataset/preProcessed_data.npy')
Y = load('/home/ram/IIITB/Sem6/VR/Project1/Ajay/VR-Project-1/dataset/lables.npy')


# Helper function for PCA Implementation

def flatten(image):
    image = np.array(image)
    return image.flatten()

print(np.array(X[0]).shape)
print(flatten(X[0]).shape[0])

DATALEN = flatten(X[0]).shape[0]
IMAGENO = len(X)

def create_img_array(image_array):
    arr = []
    for img in image_array:
        arr.append(flatten(img))
    return np.array(arr)


def compute_mean(imgarr):
    mean_face = np.zeros((DATALEN))
    for vals in imgarr:
        mean_face = np.add(mean_face,vals)
    mean_face = np.divide(mean_face,float(len(imgarr)))
    return np.squeeze(mean_face)

def normalize(imgarr,meanarr):
    normal = np.ndarray(shape=(len(imgarr), len(meanarr)))
    for i in range(len(imgarr)):
        normal[i] = np.subtract(imgarr[i],meanarr)
    return normal

def covariance(normal):
    return np.cov(normal)

def compute_eigenval(matrix):
    evals, evecs = np.linalg.eig(matrix)
    return evals,evecs

def sort_eigenval(evals,evecs, num=20):
    eig_pairs = [(np.abs(evals[i]), evecs[:,i]) for i in range(len(evals))]
    eig_pairs.sort(reverse=True)
    evecs_sort  = [eig_pairs[index][1] for index in range(len(evecs))]
    return np.array(evecs_sort[:num]).transpose()

def project_data(data, axis):
    projected = np.dot(data.transpose(),axis)
    projected = projected.transpose()
    return projected

def get_weights(projected, normal):
    w = np.array([np.dot(projected,i) for i in normal])
    return w

# Train Test Split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
print(len(x_train),len(x_test))

# PCA Implementation

img_arr = create_img_array(x_train) #create arr of images
mean = compute_mean(img_arr) #compute the mean
normal = normalize(img_arr,mean) #normalize the image
covar_matrix = covariance(normal) #compute the covariance matrix
evals,evecs = compute_eigenval(covar_matrix) #calculate the eigenvalues and eigenvectors
pc = sort_eigenval(evals, evecs, 210) #sort and get the top k eigenvectors 
projected = project_data(normal,pc) #project the data along the principal component

# Projection of Images for Feature Extraction

train_weights = get_weights(projected,normal)
train_weights.shape

test_arr = create_img_array(x_test) #create array of test images
test_normal= normalize(test_arr, mean) #normalize all of them
test_normal.shape 

test_weights = np.dot(projected, test_normal.transpose()).transpose() #project the data onto the principal component
test_weights.shape


# Training, Testing and Evaluation

output = []
for weights in test_weights:
    maxi = 0
    maxn = np.linalg.norm(train_weights[0] - weights)
    for i in range(len(train_weights)):
        diff = train_weights[i] - weights
        w = np.linalg.norm(diff)
        if(w<maxn):
            maxn = w
            maxi = i
    output.append(y_train[maxi])

accuracy_score(output, y_test)


# Graph of Accuracy vs Number of Eigen Vectors Retained

kval = [2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,400,600,800]
# kval = [1,3,10]
acc = []
i = 1
for i in kval:
    print(i)
#     kval.append(i)
    evals,evecs = compute_eigenval(covar_matrix) #calculate the eigenvalues and eigenvectors
    pc = sort_eigenval(evals, evecs, i) #sort and get the top k eigenvectors 
    projected = project_data(normal,pc) #project the data along the principal component
    train_weights = get_weights(projected,normal)
    test_weights = np.dot(projected, test_normal.transpose()).transpose() #project the data onto the principal component
    output = []
    for weights in test_weights:
        maxi = 0
        maxn = np.linalg.norm(train_weights[0] - weights)
        for j in range(len(train_weights)):
            diff = train_weights[j] - weights
            w = np.linalg.norm(diff)
            if(w<maxn):
                maxn = w
                maxi = j
        output.append(y_train[maxi])
    acc.append(accuracy_score(output, y_test))
    print(acc)
#     i = i + 200

plt.plot(kval,acc)
plt.xlabel('Number of eigenvectors', fontsize=10)
plt.ylabel('Accuracy of prediction', fontsize=10)
plt.show()


# Obtaining Top-k accuracy for k = [1,3,10] for PCA,LDA and LBP Feature Extraction methods.

face_lbph =  cv2.face.LBPHFaceRecognizer_create()
face_fisher = cv2.face.FisherFaceRecognizer_create()
face_eigen = cv2.face.EigenFaceRecognizer_create()


face_lbph.train(x_train,np.array(y_train))
face_fisher.train(x_train,np.array(y_train))
face_eigen.train(x_train,np.array(y_train))

accld = 0
acclb = 0
accei = 0

for image in range(len(x_test)):
    output1, mag1 = face_lbph.predict(x_test[image])
    output2, mag2 = face_fisher.predict(x_test[image])
    output3, mag3 = face_eigen.predict(x_test[image])
    
    if output1 == y_test[image]:
        acclb += 1.0
    if output2 == y_test[image]:
        accld += 1.0
    if output3 == y_test[image]:
        accei += 1.0


print(accei/image)
print(accld/image)
print(acclb/image)

def tester(recognizer,x_test,y_test):
    acc = [0,0,0]

    for i in range(len(x_test)):
        col = cv2.face.StandardCollector_create()
        recognizer.predict_collect(x_test[i],col)
        out = [j[0] for j in col.getResults(True)[:10]]
        
        ans = y_test[i]
        if(out[0] == ans):
            acc = np.add(acc,[1,1,1])
        elif ans in out[:3]:
            acc = np.add(acc,[0,1,1])
        elif ans in out:
            acc = np.add(acc,[0,0,1])
            
    return np.divide(acc,len(y_test))

eigen = tester(face_eigen,x_test,y_test)
print(eigen)

fisher = tester(face_fisher,x_test,y_test)
print(fisher)

lbph = tester(face_lbph,x_test,y_test)
print(lbph)
