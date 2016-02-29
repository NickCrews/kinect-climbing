import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('depth1.mov')
fgbg = cv2.createBackgroundSubtractorMOG2()

MIN_AREA = 1000

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	fgmask = fgbg.apply(frame)
	eroded = cv2.erode(fgmask, None, iterations = 2)
	dilated = cv2.dilate(eroded, None, iterations = 7)

	# Display the fgmask frame
	cv2.imshow('fgmask',fgmask)
	cv2.imshow('eroded',eroded)
	cv2.imshow('dilated',dilated)

	masked = frame.copy()
	_, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) > MIN_AREA:
			cv2.polylines(masked, [c], True, (255,255,0), 1)

	# Display original frame with mask
	cv2.imshow('img', masked)

	k = cv2.waitKey(0) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
