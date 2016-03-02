# Makefile for kinect-climbing
# Nick Crews
# 3/1/16


run:
	python testing.py

compressbgr:
	ffmpeg -f image2 -r 24 -i out/bgr/%d.png -vcodec mpeg4 -vb 20M -y data/videos/bgr.mp4

compressdepth:
	ffmpeg -f image2 -r 24 -i out/depth/%d.png -vcodec mpeg4 -vb 20M -y data/videos/depth.mp4

cleandepth:
	rm out/depth/*.png

cleanbgr:
	rm out/bgr/*.png
