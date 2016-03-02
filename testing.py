
import inputoutput as io
import processing as prc
import cv2
import numpy as np


src = io.VideoSource()
src.set_bgr_source('data/videos/bgr.mp4')
src.set_depth_source('data/videos/depth.mp4')

svr = io.VideoSaver()

cd = prc.ClimberDetector()

while (True):
	# bgr = src.get_bgr()
	depth = src.get_depth()
	# depth = mod(depth)
	mask = cd.depth2mask(depth)
	
	# svr.save_bgr(bgr)
	# mask = cd.bgr2mask(bgr)
	io.show(mask, title='final mask')
	# contours = cd.mask2contours(mask)

	# vid.show(contours)
	# cv2.waitKey(0)
	# io.show(prc.overlay(bgr, contours))
	io.show(depth,title='depth raw')
	# io.show(bgr)
	# svr.save_depth(depth)
	# svr.save_bgr(bgr)
	

	io.check_for_quit()

