# import the necessary packages
import numpy as np
import cv2

# load the image and convert it to grayscale
image = cv2.imread('barcode_01.jpg',cv2.IMREAD_COLOR)
gray = cv2.imread('barcode_01.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('Image', gray)
cv2.waitKey(0); cv2.destroyAllWindows();

# compute the gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4
ddepth = cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
cv2.imshow("gradX", gradX)
cv2.waitKey(0); cv2.destroyAllWindows();
cv2.imshow("gradY", gradY)
cv2.waitKey(0); cv2.destroyAllWindows();

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
cv2.imshow("gradient", gradient)
cv2.waitKey(0); cv2.destroyAllWindows();
gradient = cv2.convertScaleAbs(gradient)
cv2.imshow("gradient_convert", gradient)
cv2.waitKey(0); cv2.destroyAllWindows();

# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
cv2.imshow("gradient_blurred", blurred)
cv2.waitKey(0); cv2.destroyAllWindows();
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
cv2.imshow("gradient_thresh", thresh)
cv2.waitKey(0); cv2.destroyAllWindows();

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("gradient_closed", closed)
cv2.waitKey(0); cv2.destroyAllWindows();

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)
cv2.imshow("gradient_erode_dilate", closed)
cv2.waitKey(0); cv2.destroyAllWindows();

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0]
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int0(box)

# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("Image", image)
cv2.waitKey(0); cv2.destroyAllWindows();
#cv2.waitKey(0) 
