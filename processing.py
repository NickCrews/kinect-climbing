'''
processing.py
Contains all the useful methods for processing raw video feeds.
Nick Crews
2/22/16
'''
import numpy as np
import cv2
import inputoutput as io

class ClimberDetector:

	def __init__(self):
		# make instances of a background subtractor to find the climber.
		self.fgbg_depth = cv2.createBackgroundSubtractorMOG2(history = 250)
		self.fgbg_depth2 = cv2.createBackgroundSubtractorMOG2(history = 250)
		self.fgbg_depth3 = cv2.createBackgroundSubtractorMOG2(history = 250)
		self.fgbg_bgr = cv2.createBackgroundSubtractorMOG2(history = 250)

	def depth2mask(self, depth_frame):
		# innaccurate the first ~50 times called before the background subtractor learns
		# get the current mask of the foregroud (read climber)
		depth_frame = blur(depth_frame, ksize=3)
		fgmask = self.fgbg_depth.apply(depth_frame)
		# now do some processing
		blurred = cv2.medianBlur(fgmask, 5)
		return blurred

	def bgr2mask(self, bgr_frame):

		# convert to gray and blur
		gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
		gray = blur(gray, ksize=3)
		# innaccurate the first ~50 times called before the background subtractor learns
		# get the current mask of the foregroud (read climber)
		fgmask = self.fgbg_bgr.apply(bgr_frame)
		blurred = cv2.medianBlur(fgmask, 5)

		return blurred
		# # now do some processing
		# # first fill in holes
		# dilated = cv2.dilate(fgmask, None, iterations = 2)
		# # now get rid of everything except the largest blob
		# _, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		# largest = max(cnts, key=lambda c:cv2.contourArea(c))
		# mask = np.zeros_like(bgr_frame)
		# cv2.drawContours(mask, largest, 0, 255, -1)
		# return mask

	def mask2contours(self, mask):
		# whats the smallest blob which we can call a climber?
		MIN_AREA = 10
		# get the contours
		_, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		return [c for c in cnts if cv2.contourArea(c) > MIN_AREA]

def pretty_depth(frame):
	assert frame.dtype == np.uint32
	np.clip(depth, 0, 2**10 - 1, depth)
	depth >>= 2
	depth = depth.astype(np.uint8)
	return depth

def blur(img, ksize=5, sigmaX = 0):
	return cv2.GaussianBlur(img, (ksize,ksize), sigmaX)

def overlay(img, contours):
	'''Returns acopy of image with contour drawn over the top'''
	result = img.copy()
	cv2.polylines(result, contours, True, (255,255,0), 1)
	return result




