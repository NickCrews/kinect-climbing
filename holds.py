import cv2
import numpy as np
from matplotlib import pyplot as plt
import inputoutput as io

def resize(img, newWidth):
	r = float(newWidth) / img.shape[1]
	dim = (newWidth, int(img.shape[0] * r))
	return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def blur(img):
	return cv2.GaussianBlur(img,(5,5),0)

def auto_canny(image, sigma=0.5):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def lap(img):
	return cv2.Laplacian(img,cv2.CV_64F)

def overlay(img, canny, MIN_AREA = 35, MAX_AREA = 5000):
	masked = img.copy()
	_, cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	hulls = [cv2.convexHull(c) for c in cnts]
	for h in hulls:
			# if the contour is too small, ignore it
			area = cv2.contourArea(h)
			if area>MIN_AREA and area<MAX_AREA:
				cv2.polylines(masked, [h], True, (255,255,0), 1)
	return masked

def main():	
	src = io.VideoSource()
	src.set_bgr_source('out/bgr1')
	while (True):
		# get the current bgr frame and Mean Shift Filter it
		bgr = src.get_bgr()
		shifted = cv2.pyrMeanShiftFiltering(bgr, 12, 20)
		io.show(shifted, 'shifted')

		# perform Canny edge detection on the MSF image
		can = auto_canny(shifted)
		# overlay the detected edges
		overlay = bgr.copy()
		overlay[np.where(can)] = (255,255,255)

		# show the unadultered bgr frame and the overlayed
		io.show(bgr, 'bgr')
		io.show(overlay, 'overlay')

		if io.pause():
			break

if __name__ == '__main__':
	main()

