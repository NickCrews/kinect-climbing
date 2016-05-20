'''
processing.py
Contains all the useful methods for processing raw video feeds.
Nick Crews
2/22/16
'''
import numpy as np
import cv2
import inputoutput as io
from skimage.morphology import skeletonize


_depth_pts = np.float32([[92, 76], [537, 92], [605, 469], [78, 461]])
_bgr_pts = np.float32([[121, 75], [514, 97], [567, 430], [107, 415]])

_M_bgr2depth = cv2.getPerspectiveTransform(_bgr_pts, _depth_pts)
_M_depth2bgr = cv2.getPerspectiveTransform(_depth_pts, _bgr_pts)

class ClimberDetector:

	def __init__(self):
		# make instances of a background subtractor to find the climber.
		d_hist = 100
		self.d_rate = 1.0/d_hist

		self.fgbg_depth = cv2.createBackgroundSubtractorMOG2(history = d_hist, varThreshold=1)
		self.fgbg_depth.setBackgroundRatio(.1)
		self.fgbg_depth.setNMixtures(10)
		self.fgbg_depth.setVarThreshold(24)
		
		bgr_hist = 500
		self.bgr_rate = 1.0/bgr_hist
		self.fgbg_bgr = cv2.createBackgroundSubtractorMOG2(history = bgr_hist)
		self.shadowVal = self.fgbg_bgr.getShadowValue()

	def depth2mask(self, depth_frame):
		# innaccurate the first ~50 times called before the background subtractor learns
		# get the current mask of the foreground (read climber)
		depth_frame = blur(depth_frame, 7)
		fgmask = self.fgbg_depth.apply(depth_frame,learningRate=self.d_rate)
		# now do some processing
		blurred = cv2.medianBlur(fgmask, 3)
		return blurred

	def bgr2mask(self, bgr_frame):
		# innaccurate the first ~50 times called before the background subtractor learns
		# get the current mask of the foregroud (read climber)
		fgmask = self.fgbg_bgr.apply(bgr_frame, learningRate=self.bgr_rate)
		return fgmask

	def mask2contours(self, mask):
		
		# get the contours
		_, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		return [c for c in cnts if cv2.contourArea(c) > self.MIN_AREA]

	def centerOfMask(self, mask):
		_, noShadows = cv2.threshold(mask,self.shadowVal,255,cv2.THRESH_BINARY)

		# eroded = cv2.erode(noShadows,np.ones((5,5),np.uint8))
		blurred = cv2.medianBlur(noShadows, 3)
		dilated = cv2.dilate(blurred, np.ones((5,5), np.uint8))

		# io.show(blurred, 'blurredsssss')
		# io.show(dilated, 'dilated')

		# get the largest contour in the mask
		_, cnts, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		if len(cnts)==0:
			return None
		largest_contour = max(cnts, key = lambda c: cv2.contourArea(c))

		# get moments of the contours
		M = cv2.moments(largest_contour)
		# get the center
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])

		return (cX, cY)

	def erodedMask(self,mask):
		ns = self.noShadows(mask)
		eroded = cv2.erode(ns ,np.ones((5,5),np.uint8), iterations=2)
		return eroded

	def noShadows(self,mask):
		ret, result = cv2.threshold(mask,self.shadowVal,255,cv2.THRESH_BINARY)
		return result


def largest_blob(frame):
	'''Returns the largest contiguous blob of pixels'''
	# find the largest contour
	_, cnts, _ = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	largest_contour = max(cnts, key = lambda c: cv2.contourArea(c))
	# initialize our result mask
	result = np.zeros_like(frame, dtype = np.uint8)
	# fill in the largest contour
	cv2.drawContours(result, [largest_contour], 0, (255), -1)
	return result

def pretty_depth(frame):
	'''Assumes a very raw 16 bit depth image from the kinect. Converts to an 8 bit image, keeping as much contrast as possible.'''
	frame = frame.copy()
	errs = np.where(frame==2047)
	# constrain the frame to 0-255
	frame = np.clip(frame, 768, 1023)
	frame = frame - 768
	frame = np.uint8(frame)
	# ok, now we have a uint8 object to deal with


	# now let's fix all the errors using inpainting
	mask = np.zeros(frame.shape, dtype=np.uint8)
	mask[errs] = 1
	inpainted = cv2.inpaint(frame,mask,2,cv2.INPAINT_TELEA)

	# there are also some errors around the edge of the image which dont get dealt with
	cropped = inpainted[1:-1, 1:-1]
	padded = cv2.copyMakeBorder(cropped,1,1,1,1,cv2.BORDER_REPLICATE)

	return padded

def blur(img, *kargs):
	ksize, sigmaX = 5, 0
	if len(kargs) == 1:
		ksize = kargs[0]
	elif len(kargs) == 2:
		ksize, sigmaX = kargs
	return cv2.GaussianBlur(img, (ksize,ksize), sigmaX)

def depth2bgr(depth_frame):
	"""Maps a depth frame to the BGR frame"""
	desired_dimensions = (640, 480)
	return cv2.warpPerspective(depth_frame,_M_depth2bgr,desired_dimensions)

def bgr2depth(bgr_frame):
	"""Maps a BGR frame to the depthframe"""
	desired_dimensions = (640, 480)
	return cv2.warpPerspective(depth_frame,_M_bgr2depth,desired_dimensions)


