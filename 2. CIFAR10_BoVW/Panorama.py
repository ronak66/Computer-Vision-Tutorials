import numpy as np
import imutils
import cv2

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)
    # There's a big difference in how OpenCV 2.4 and OpenCV 3 deals with keypoint detection and local invariant descriptors
    
    def stitch(self, images, ratio=0.75, reproThresh=4.0,showMatches=False):
        # Unpack the Two images, detect and extract local invariant descriptors
        (B,A) = images
        (kpsA, featuresA) = self.detectDescribe(A)
        (kpsB, featuresB) = self.detectDescribe(B)
        
        # Match the keypoints across the two images
        M = self.matchKeypoints(kpsA,kpsB,featuresA,featuresB,ratio,reproThresh)
        
        # If match is None, not enough matched keypoints for a panorama
        if M is None:
            return None
        
        # Apply a perspective warp to stitch the images together
        (matches, H, status) = M
        result = cv2.warpPerspective(A, H, 
                                     (A.shape[1]+B.shape[1],A.shape[0]))
        result[0:B.shape[0],0:B.shape[1]] = B
        
        # Keypoint matches should be visualized or not
        if showMatches:
            vis = self.drawMatches(A,B,kpsA,kpsB,matches,status)
            return (result,vis)
        return result

    def detectDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.isv3:
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps,features) = descriptor.detectAndCompute(image, None)
        
        else:
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)
            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)
        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keyp
        return (kps,features)
    
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
	# matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)
            # otherwise, n
        return None
        
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # return the visualization
        return vis

if __name__ == '__main__':
	# load the two images and resize them to have a width of 400 pixels
	# (for faster processing)
	imageA = cv2.imread("secondPic1.jpg")
	imageB = cv2.imread("secondPic2.jpg")
	imageA = imutils.resize(imageA, width=400)
	imageB = imutils.resize(imageB, width=400)
	# stitch the images together to create a panorama
	stitcher = Stitcher()
	(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
	# show the images
	cv2.imshow("Image A", imageA)
	cv2.imshow("Image B", imageB)
	cv2.imshow("Keypoint Matches", vis)
	cv2.imshow("Result", result)
	cv2.waitKey(0)



