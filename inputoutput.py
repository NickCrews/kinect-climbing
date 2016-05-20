'''
inputoutput.py
Contains all the low level input-output interfaces with the kinect device, saved recordings, or screen.
Nick Crews
2/22/16
'''


import processing as prc
import freenect
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


class VideoSource:

	def __init__(self, bgr = 'KINECT', depth = 'KINECT', ir = 'KINECT', img_format = '%1d.png'):

		# What are the sources for the three feeds? Can be the String 'KINECT', in which case will try to read from connected camera, or can be a cv2.VidoCapture object to read from
		self.bgr = bgr
		self.depth = depth
		self.ir = ir

		# what kind of images are we reading
		self.image_format = img_format

	def get_bgr(self):
		if self.bgr == 'KINECT':
			frame, timestamp = freenect.sync_get_video()
			bgr = frame[:, :, ::-1]  # RGB -> BGR
			return bgr
		else:
			ret, img = self.bgr.read()
			return img

	def get_depth(self):
		if self.depth == 'KINECT':
			frame, timestamp = freenect.sync_get_depth()
			return frame
		else:
			ret, img = self.depth.read()
			return img

	def get_ir(self):
		if self.ir == 'KINECT':
			frame, timestamp = freenect.sync_get_video(freenect.VIDEO_IR_10BIT)
			return frame
		else:
			ret, img = self.ir.read()
			return img
	
	def set_bgr_source(self, directorypath):
		if directorypath == 'KINECT':
			self.bgr = 'KINECT'
		else:
			self.bgr = cv2.VideoCapture(directorypath + os.sep + self.image_format)

	def set_depth_source(self, directorypath):
		if directorypath == 'KINECT':
			self.depth = 'KINECT'
		else:
			self.depth = cv2.VideoCapture(directorypath + os.sep + self.image_format)

	def set_ir_source(self, directorypath):
		if directorypath == 'KINECT':
			self.ir = 'KINECT'
		else:
			self.ir = cv2.VideoCapture(directorypath + os.sep + self.image_format)

class VideoSaver:
	'''Class which handles writing video feeds'''

	def __init__(self, bgr='bgr', depth='depth', ir='ir', ext='.png'):

		self.bgr_path = 'out' + os.sep + bgr
		self.depth_path = 'out' + os.sep + depth
		self.ir_path = 'out' + os.sep + ir

		self._bgr_count = 0
		self._depth_count = 0
		self._ir_count = 0

		self.extension = ext

		# create the folders if they don't exist
		for path in [self.bgr_path, self.depth_path, self.ir_path]:
			if not os.path.exists(path):
			    os.makedirs(path)
			else:
				for the_file in os.listdir(path):
				    file_path = os.path.join(path, the_file)
				    try:
				        if os.path.isfile(file_path):
				            os.unlink(file_path)
				    except Exception as e:
				        print e

	def clean_bgr(self):
		for f in os.listdir(self.bgr_path):
			os.remove(f)

	def clean_depth(self):
		for f in os.listdir(self.depth_path):
			os.remove(f)

	def clean_ir(self):
		for f in os.listdir(self.ir_path):
			os.remove(f)

	def save_bgr(self, frame):
		path = self.bgr_path + os.sep + str(self._bgr_count) + self.extension
		cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
		self._bgr_count += 1

	def save_depth(self, frame):
		path = self.depth_path + os.sep + str(self._depth_count) + self.extension
		cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
		self._depth_count += 1

	def save_ir(self, frame):
		path = self.ir_path + os.sep + str(self._ir_count) + self.extension
		cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
		self._ir_count += 1

def show(img, *kargs):
	title = kargs[0] if len(kargs)==1 else 'DEFAULT'
	cv2.imshow(title, img)

def check_for_quit():
	'''If the user recently pressed q, return True. Otherwise return False immediately'''
	if cv2.waitKey(1) == ord('q'):
		return True
	return False

def pause():
	'''Wait for the user to press something. if 'q', return True. Else False'''
	if cv2.waitKey(0) == ord('q'):
		return True
	return False

def histogram(img, *kargs, **kwargs):
	title = kargs[0] if len(kargs)==1 else 'DEFAULT'

	plt.clf()
	plt.hist(img.ravel(), bins = 56, range=(0, np.max(img)))
	plt.draw()
	plt.pause(.001)






