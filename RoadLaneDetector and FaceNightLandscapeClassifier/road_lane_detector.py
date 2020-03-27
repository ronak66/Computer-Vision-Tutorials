import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

class RoadLaneDetector():

    def grayscale(self,img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def canny(self,img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self,img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self,img, vertices):
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            channel_count = img.shape[2]  
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
    
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def draw_lines(self,img, lines, color=[255, 0, 0], thickness=2):
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def hough_lines(self,img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        draw_lines(line_img, lines)
        return line_img


    def weighted_img(self,img, initial_img, α=0.8, β=1., λ=0.):
        return cv2.addWeighted(initial_img, α, img, β, λ)

    def process_frame(self,image):
        global first_frame

        gray_image = self.grayscale(image)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        lower_yellow = np.array([20, 100, 100], dtype = "uint8")
        upper_yellow = np.array([30, 255, 255], dtype="uint8")

        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(gray_image, 200, 255)
        
        mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
        mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

        kernel_size = 5
        gauss_gray = self.gaussian_blur(mask_yw_image,kernel_size)

        low_threshold = 50
        high_threshold = 150
        canny_edges = self.canny(gauss_gray,low_threshold,high_threshold)

        imshape = image.shape
        lower_left = [imshape[1]/9,imshape[0]]
        lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
        top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
        top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
        vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
        roi_image = region_of_interest(canny_edges, vertices)

        rho = 2
        theta = np.pi/180

        threshold = 20
        min_line_len = 50
        max_line_gap = 200

        line_image = self.hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
        result = self.weighted_img(line_image, image, α=0.8, β=1., λ=0.)
        return result

    def show_image(self,img):
        cv2.imshow("output",img)
        cv2.imwrite('./static/whitening.png',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    def image_segmenting(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        plt.imshow(thresh, cmap = "binary")
        vectorized = img.reshape((-1,3))
        vectorized = np.float32(vectorized)
        print(img.shape, vectorized.shape)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        attempts=10
        ret, label, centers = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        res = centers[label.flatten()]
        result_image = res.reshape((img.shape))
        figure_size = 15
        plt.figure(figsize=(figure_size,figure_size))
        plt.subplot(1,2,1),plt.imshow(img[:, :, [2, 1, 0]])
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(1,2,2),plt.imshow(result_image)
        plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
        plt.show()

if __name__ == '__main__':
    image = mpimg.imread('assets/img6.jpeg')
    rld = RoadLaneDetector()
    # processed = rld.process_frame(image)
    # rld.show_image(processed)
    rld.image_segmenting(image)