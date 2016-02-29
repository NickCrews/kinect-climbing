# Standard imports
import cv2
import numpy as np;

def resize(img, newWidth):
	r = float(newWidth) / img.shape[1]
	dim = (newWidth, int(img.shape[0] * r))
	return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# Read image
orig = cv2.imread('data/1.jpg', cv2.IMREAD_COLOR)
resized = resize(orig, 700)
l, a, b = cv2.split(cv2.cvtColor(resized,cv2.COLOR_RGB2Lab))
a = cv2.GaussianBlur(a,(5,5),0)

# Setup SimpleBlobDetector parameters.
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 1;
params.maxThreshold = 1000;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 200
params.maxArea = 5000
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.01
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87
 
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01
 
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(a)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(resized, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("a", a)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

