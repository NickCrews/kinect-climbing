'''
video.py
A Class which contains all the low level interfaces with the kinect device or saved recordings
Nick Crews
2/22/16
'''

import freenect
import numpy as np
import cv2
import os

class VideoSource:

	def __init__(self, bgr = 'KINECT', depth = 'KINECT', ir = 'KINECT'):
		# What are the sources for the three feeds? Can be the String 'KINECT', in which case will try to read from connected camera, or can be a cv2.VidoCapture object to read from
		self.bgr = bgr
		self.depth = depth
		self.ir = ir

		# The index of the active Kinect. We assume there's only one kinect connect
		self.DEVICE_INDEX = 0

	def get_bgr(self):
		if self.bgr == 'KINECT':
			return self._k_bgr()
		else:
			return self.bgr.read()[1]

	def get_depth(self):
		if self.depth == 'KINECT':
			return self._k_depth()
		else:
			return self.depth.read()[1]

	def get_ir(self):
		if self.ir == 'KINECT':
			return self._k_ir()
		else:
			return self.ir.read()[1]
		
	def set_bgr_source(self, filepath):
		self.bgr = cv2.VideoCapture(filepath)

	def set_depth_source(self, filepath):
		self.depth = cv2.VideoCapture(filepath)

	def set_ir_source(self, filepath):
		self.ir = cv2.VideoCapture(filepath)

	def _k_bgr(self):
		'''Get the current bgr frame from the Kinect'''
		try:
			video = freenect.sync_get_video(self.self.DEVICE_INDEX_INDEX)[0]
		except:
			video = freenect.sync_get_video(self.DEVICE_INDEX)[0]
		bgr = video[:, :, ::-1]  # RGB -> BGR
		return bgr

	def _k_depth(self):
		'''Get the current depth frame from the Kinect'''
		try:
			depth = freenect.sync_get_depth(self.DEVICE_INDEX)[0]
		except :
			depth = freenect.sync_get_depth(self.DEVICE_INDEX)[0]
		np.clip(depth, 0, 2**10 - 1, depth)
		depth >>= 2
		depth = depth.astype(np.uint8)
		return depth

	def _k_ir(self):
		'''Get the current ir frame from the Kinect'''
		try:
			frame = freenect.sync_get_video(self.DEVICE_INDEX, freenect.VIDEO_IR_10BIT)[0]
		except TypeError:
			frame = freenect.sync_get_video(self.DEVICE_INDEX, freenect.VIDEO_IR_10BIT)[0]
		np.clip(frame, 0, 2**10 - 1, frame)
		frame >>= 2
		frame = frame.astype(np.uint8)
		return frame

class VideoSaver:
	'''Class which handles writing video feeds'''

	def __init__(self):

		self.bgr_path = 'out' + os.sep + 'bgr'
		self.depth_path = 'out' + os.sep + 'depth'
		self.ir_path = 'out' + os.sep + 'ir'

		self._bgr_count = 0
		self._depth_count = 0
		self._ir_count = 0

		self.extension = '.png'

	def save_bgr(self, frame):
		path = self.bgr_path + os.sep + str(self._bgr_count) + self.extension
		cv2.imwrite(path, frame)
		self._bgr_count += 1

	def save_depth(self, frame):
		path = self.depth_path + os.sep + str(self._depth_count) + self.extension
		cv2.imwrite(path, frame)
		self._depth_count += 1

	def save_ir(self, frame):
		path = self.ir_path + os.sep + str(self._ir_count) + self.extension
		cv2.imwrite(path, frame)
		self._ir_count += 1

def show(img, title = 'FRAME'):
	cv2.imshow(title, img)






