import cv2
import numpy as np
import matplotlib.pyplot as plt

class ComputerVision:

    def __init__(self,image_path):
        self.img = cv2.imread(image_path)

    def whitening(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        n, m = gray.shape
        mean = np.mean(gray)
        var = np.var(gray)
        sigma = var**0.5
        gray = (gray - mean)/sigma
        return gray

    def history_equalization(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        n, m = gray.shape
        hist = np.histogram(gray, bins=range(256))[0]
        plt.plot(hist)
        plt.show()
        cummulative_sum=[]
        cummulative_sum.append(hist[0])       
        for i in range(1,len(hist)):
            cummulative_sum.append(cummulative_sum[i-1]+hist[i])
        plt.plot(cummulative_sum)
        plt.show()
        for i in range(len(gray)):
            for j in range(len(gray[i])):
                gray[i][j] = (255*cummulative_sum[int(gray[i][j])])/(n*m)
        return gray
        

    @staticmethod
    def show_image(img):
        cv2.imshow("output",img)
        cv2.imwrite('./static/whitening.png',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    @staticmethod
    def plt_image(img,cmap):
        plt.imshow(img, cmap=cmap)
        plt.show()
        return
