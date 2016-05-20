
import inputoutput as io
import processing as prc
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

	src = io.VideoSource()
	src.set_depth_source('out/depth3')
	src.set_bgr_source('out/bgr3')


	svr = io.VideoSaver()

	cd = prc.ClimberDetector()
	cd2 = prc.ClimberDetector()

	clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(1,1))

	while (True):
		# get the current depth frame
		depth = src.get_depth()
		io.show(depth, 'depth')
		# normalize and fix it
		pretty = prc.pretty_depth(depth)
		io.show(pretty,'pretty')
		# blur it, clahe it
		blur = cv2.medianBlur(pretty, 7)
		cl = clahe.apply(blur)
		io.show(cl, 'after clahe')
		# apply the clahe'd depth image to the background subtractor
		depthmask = cd.depth2mask(cl)

		# get the current rgb (actually bgr) frame, and overlay the climber mask on it
		bgr = src.get_bgr()
		bgr[prc.depth2bgr(depthmask)>0] = (0, 255, 0)
		io.show(bgr, 'bgr')

		# wait for the user to press something. If it was 'q', then quit
		if io.pause():
			break

if __name__ == '__main__':
	main()
