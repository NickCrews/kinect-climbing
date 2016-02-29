
import video as vid
import processing as prc
import cv2



src = vid.VideoSource()
src.set_bgr_source('data/videos/rgb.mp4')
src.set_depth_source('data/videos/depth1.mov')

svr = vid.VideoSaver()

cd = prc.ClimberDetector()

while (True):
	print 'loop!'
	# bgr = src.get_bgr()
	# bgr = cv2.resize(bgr, (0,0), fx=2, fy=2)
	depth = src.get_depth()
	mask = cd.depth2mask(depth)
	# vid.show(bgr)
	# svr.save_bgr(bgr)
	# mask = cd.bgr2mask(bgr)
	# vid.show(mask)
	# cv2.waitKey(0)
	contours = cd.mask2contours(mask)

	# vid.show(contours)
	# cv2.waitKey(0)
	vid.show(prc.overlay(depth, contours))
	cv2.waitKey(1)
