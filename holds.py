import cv2
import numpy as np
from matplotlib import pyplot as plt
import inspect

def varName(var):
    lcls = inspect.stack()[2][0].f_locals
    for name in lcls:
        if id(var) == id(lcls[name]):
            return name
    return None

def resize(img, newWidth):
	r = float(newWidth) / img.shape[1]
	dim = (newWidth, int(img.shape[0] * r))
	return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def blur(img):
	return cv2.GaussianBlur(img,(5,5),0)

def auto_canny(image, sigma=0.5):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def show(img,name):
	cv2.imshow(name, img)

def lap(img):
	return cv2.Laplacian(img,cv2.CV_64F)

def doAll(listOfItems, action):
	return [action(item) for item in listOfItems]

def overlay(img, canny, MIN_AREA = 35, MAX_AREA = 5000):
	masked = img.copy()
	_, cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	hulls = [cv2.convexHull(c) for c in cnts]
	for h in hulls:
			# if the contour is too small, ignore it
			area = cv2.contourArea(h)
			if area>MIN_AREA and area<MAX_AREA:
				cv2.polylines(masked, [h], True, (255,255,0), 1)
	return masked

orig = cv2.imread('data/1.jpg', cv2.IMREAD_COLOR)
orig = resize(orig, 900)

gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(orig,cv2.COLOR_RGB2Lab)
ycrcb = cv2.cvtColor(orig,cv2.COLOR_RGB2YCrCb)
hls = cv2.cvtColor(orig,cv2.COLOR_RGB2HLS)
xyz = cv2.cvtColor(orig,cv2.COLOR_RGB2XYZ)
luv = cv2.cvtColor(orig,cv2.COLOR_RGB2Luv)

r, g, b = cv2.split(orig)
h, s, v = cv2.split(hsv)
l, a, b = cv2.split(lab)
y, cr, cb = cv2.split(ycrcb)
h2, l2, s2 = cv2.split(hls)
x, y, z = cv2.split(xyz)
l3, u3, v3 = cv2.split(luv)

# weCare = [a, b, cr, cb, u3, s2, h2]

# doAll(doAll(weCare, lap), show)

a_blur = blur(a)
a_lap = lap(a_blur)
rescaled = cv2.equalizeHist(cv2.convertScaleAbs(a_lap))

# show(a_blur, 'a_blur')
# show(a_lap,'a_lap')
# show(rescaled, 'rescaled')

# print(a_lap.dtype)
# print(np.max(a_lap))
# print(np.min(a_lap))

# print(rescaled.dtype)
# print(np.max(rescaled))
# print(np.min(rescaled))

# show(auto_canny(rescaled), 'automatic')


# show(lap(blur(a)), 'a')
# show(lap(blur(b)), 'b')
# show(lap(blur(cr)), 'cr')
# show(cv2.convertScaleAbs(lap(blur(cb))), 'cb')


# plt.hist(a_lap.ravel(),64,[-64,64], label = "a")
# # plt.hist(lap(blur(a)).ravel(),64,[0,64], label = "ablurred")
# # plt.legend()
# plt.show()

# cv2.imshow('bgr',bgr)
# cv2.imshow('gray',gray)

# cv2.imshow('b', b)
# cv2.imshow('g', g)
# cv2.imshow('r', r)

# cv2.imshow('h', h)
# cv2.imshow('s', s)
# cv2.imshow('v', v)

# cv2.imshow('l', l)
# cv2.imshow('a', a)
# cv2.imshow('b', b)

# cv2.imshow('y', y)
# cv2.imshow('cr', cr)
# cv2.imshow('cb', cb)

# cv2.imshow('h2', h2)
# cv2.imshow('l2', l2)
# cv2.imshow('s2', s2)

# cv2.imshow('x', x)
# cv2.imshow('y', y)
# cv2.imshow('z', z)

# cv2.imshow('l3', l3)
# cv2.imshow('u3', u3)
# cv2.imshow('v3', v3)

# plt.hist(a.ravel(),64,[0,256], label = "a")
# # plt.hist(aq.ravel(),16,[0,256], label = "aq")
# plt.legend()
# plt.show()

ac = cv2.Canny(blur(a), 30, 40)/6
bc = cv2.Canny(blur(b), 30, 40)/6
vc = cv2.Canny(blur(v), 60, 70)/6
crc = cv2.Canny(blur(cr), 20, 40)/6
cbc = cv2.Canny(blur(cb), 30, 70)/6
v3c = cv2.Canny(blur(v3), 30, 70)/6

added = ac + bc + vc + crc + cbc + v3c

plt.hist(added.ravel(),64,[0,256])
plt.show()

filtered = cv2.threshold(added, 50, 255, cv2.THRESH_BINARY)[1]
# dilated = cv2.dilate(filtered, None, iterations = 0)
# eroded = cv2.erode(filtered, None, iterations = 0)

cv2.imshow('ac', overlay(a, ac))
cv2.imshow('bc', overlay(b, bc))
cv2.imshow('vc', overlay(v, vc))
cv2.imshow('crc', overlay(cr, crc))
cv2.imshow('cbc', overlay(cb, cbc))
cv2.imshow('v3c', overlay(v3, v3c))

cv2.imshow('added', added)
cv2.imshow('filtered', filtered)
# cv2.imshow('dilated', dilated)
# cv2.imshow('eroded', eroded)
cv2.imshow('overlayed final', overlay(orig, filtered))




cv2.waitKey(0)
cv2.destroyAllWindows()

