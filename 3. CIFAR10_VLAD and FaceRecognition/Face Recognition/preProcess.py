import numpy as np
import cv2
import os
import math
from numpy import save

def load_dataset(path, classes) :
    class_images = []
    labels = []
    for cls in classes :
        cls_imgs = []
        img_names = os.listdir(path + cls + "/")
        for img_name in img_names :
                try :
                    img = cv2.imread(path + cls + "/" + img_name)
                    cls_imgs.append(img)
                    labels.append(cls)
                except Exception as e :
                    pass
        class_images.append(cls_imgs)
    return np.array(class_images), labels

def convertToGrayScale(dataset) :
    gray_dataset = []
    for x in dataset :
        gray_class = []
        for y in x :
            gray_img = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
            gray_class.append(gray_img)
        gray_dataset.append(gray_class)
    return np.array(gray_dataset)

def detectEyes(dataset, labels) :
    #Load the classifiers
    face_cascade = cv2.CascadeClassifier('/home/ajayrr/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier('/home/ajayrr/opencv/data/haarcascades_cuda/haarcascade_eye_tree_eyeglasses.xml')
    faces = []
    num_undetected_faces = 0
    num_faces = 0
    num_eyes = 0
    num_undetected_eyes = 0

    index = 0

    for c in dataset :
        for i in c :

            num_faces += 1
            #Detect the face
            faces_detected = face_cascade.detectMultiScale(i, scaleFactor=1.1, minNeighbors=4)
            if (len(faces_detected) != 0 ) :
                (x, y, w, h) = faces_detected[0]


                #Detect the eyes
                eyes = eyes_cascade.detectMultiScale(i[y:y+h, x:x+w], scaleFactor = 1.1, minNeighbors = 5)

                if len(eyes) > 0 :

                    (ex1, ey1, ew1, eh1) = eyes[0]
                    (cx1, cy1) = (ex1 + ew1, ey1 + eh1)
                    for m in range(1, len(eyes)) :
                        (ex2, ey2, ew2, eh2) = eyes[m]
                        (cx2, cy2) = (ex2 + ew2, ey2 + eh2)
                        slope = (cy2 - cy1)/(cx2 - cx1)
                        angle = 180*(math.atan(slope))/np.pi
                        if abs(angle) <= 10  and (cx2 - cx1 >= 50):
                            break
                        if m == len(eyes) - 1 :
                            (cx2, cy2) = (cx1, cy1)
                    if len(eyes) == 1 :
                        (cx2, cy2) = (cx1, cy1)

                    if cx2 != cx1 :
                        # print(angle)
                        rows, cols = i.shape[:2]
                        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                        i = cv2.warpAffine(i, M, (cols,rows))
                        if abs(angle) > 20 :
                            cv2.imshow("Original Image", i)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                        #Detect the faces again
                        img_faces_detected = face_cascade.detectMultiScale(i, scaleFactor=1.1, minNeighbors=3)

                        if len(img_faces_detected) == 0 :
                            cv2.imshow("Original Image", i)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        (x, y, w, h) = img_faces_detected[0]
                        p = 0
                        i = i[y-p+1:y+h+p, x-p+1:x+w+p]
                        i = cv2.resize(i, (256, 256), interpolation=cv2.INTER_LINEAR)
                        faces.append(i)
                    else :
                        p = 0
                        i = i[y-p+1:y+h+p, x-p+1:x+w+p]
                        i = cv2.resize(i, (256, 256), interpolation=cv2.INTER_LINEAR)
                        faces.append(i)
                else :
                    num_undetected_eyes += 1
                    p = 0
                    i = i[y-p+1:y+h+p, x-p+1:x+w+p]
                    i = cv2.resize(i, (256, 256), interpolation=cv2.INTER_LINEAR)
                    faces.append(i)
            else :
                num_undetected_faces += 1
                value = labels.pop(index)


    print("total number of faces : ", num_faces)
    print("undetected faces : ", num_undetected_faces)
    print("undetected eyes : ", num_undetected_eyes)
    return np.array(faces), np.array(labels)

def preProcess(dataset, labels) :
    gray_dataset = convertToGrayScale(dataset)
    preProcessed_dataset, preProcessed_labels = detectEyes(gray_dataset, labels)
    print('PreProcessing Complete')
    return preProcessed_dataset, preProcessed_labels

if __name__ == "__main__" :

    DATADIR = "/home/ajayrr/Semester6/vr/projects/p1/face_dataset/"
    classes = os.listdir("/home/ajayrr/Semester6/vr/projects/p1/face_dataset")

    dataset, labels = load_dataset(DATADIR, classes)


    preprocessed_data, preprocessed_labels = preProcess(dataset, labels)
    print(preprocessed_data.shape)
    save('preProcessed_data.npy', preprocessed_data)
    save('lables.npy', preprocessed_labels)
