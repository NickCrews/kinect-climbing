
import inputoutput as io
import processing as prc
import cv2
import numpy as np
import sys

if len(sys.argv) != 3:
	print 'usage: python record.py <depthdirectory> <bgrdirectory>'
	exit()


src = io.VideoSource()
# src.set_bgr_source('data/videos/bgr2.mp4')
# src.set_depth_source('out/depth1')

depthpath, bgrpath = sys.argv[1], sys.argv[2]
svr = io.VideoSaver(depth = depthpath, bgr = bgrpath)



while (True):
	depth = src.get_depth()
	bgr = src.get_bgr()


	io.show(depth,'depth')
	io.show(bgr,'bgr')


	# pretty = prc.pretty_depth(depth)
	# io.show(pretty,'pretty')

	svr.save_depth(depth)
	svr.save_bgr(bgr)

	# cutoff = pretty & 255<<8
	# io.show(cutoff,'cutoff')
	
	# if io.pause():
	# 	break
	if io.check_for_quit():
		break

