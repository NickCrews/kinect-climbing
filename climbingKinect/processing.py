'''
processing.py
Contains all the useful helper methods for processing raw video feeds.
Nick Crews
2/22/16
'''
import numpy as np
import cv2

class ClimberDetector:

	def __init__(self):
		# make an instance of a background subtractor to find the climber.
		self.fgbg_depth = cv2.createBackgroundSubtractorMOG2()
		self.fgbg_bgr = cv2.createBackgroundSubtractorMOG2()

	def depth2mask(self, depth_frame):
		# innaccurate the first ~50 times called before the background subtractor learns
		# get the current mask of the foregroud (read climber)
		fgmask = self.fgbg_depth.apply(depth_frame)
		# now do some processing
		return cv2.medianBlur(fgmask, 5)

		# first fill in holes
		dilated = cv2.dilate(fgmask, None, iterations = 2)
		# now get rid of everything except the largest blob
		_, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		largest = max(cnts, key=lambda c:cv2.contourArea(c))
		mask = np.zeros_like(depth_frame)
		cv2.drawContours(mask, largest, 0, 255, -1)
		return mask

	def bgr2mask(self, bgr_frame):

		# convert to gray
		gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
		# innaccurate the first ~50 times called before the background subtractor learns
		# get the current mask of the foregroud (read climber)
		fgmask = self.fgbg_bgr.apply(bgr_frame)
		
		# median_blurred3 = cv2.medianBlur(fgmask, 3)
		median_blurred5 = cv2.medianBlur(fgmask, 5)
		# cv2.imshow('basic', fgmask)
		# cv2.imshow('median_blurred3', median_blurred3)
		cv2.imshow('median_blurred5', median_blurred5)
		return median_blurred5
		# now do some processing
		# first fill in holes
		dilated = cv2.dilate(fgmask, None, iterations = 2)
		# now get rid of everything except the largest blob
		_, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		largest = max(cnts, key=lambda c:cv2.contourArea(c))
		mask = np.zeros_like(bgr_frame)
		cv2.drawContours(mask, largest, 0, 255, -1)
		return mask

	def mask2contours(self, mask):
		# whats the smallest blob which we can call a climber?
		MIN_AREA = 100
		# get the contours
		_, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		return [c for c in cnts if cv2.contourArea(c) > MIN_AREA]


def overlay(img, contours):
	'''Returns acopy of image with contour drawn over the top'''
	result = img.copy()
	cv2.polylines(result, contours, True, (255,255,0), 1)
	return result




