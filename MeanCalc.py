#!/usr/bin/env python3
import imutils
import cv2
import numpy as np
from scipy.stats import norm

image = cv2.imread("WIN_20230328_12_52_03_Pro.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Noise reduction
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# thresholding
mean, std = norm.fit(blurred)
thresh_min_value = int(mean + 3.6*std)
thresh = cv2.threshold(blurred, thresh_min_value, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    # Moments
    M = cv2.moments(c)
    # Area of contour
    area = M["m00"]

    # Centroid of contour
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Perimeter
    perimeter = cv2.arcLength(c, True)

    print(f'{area=}')
    print(f'{perimeter=}')

    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(c, epsilon, True)
    print(f'{approx=}')

    # Create blank image
    blank_image = np.zeros(gray.shape, np.uint8)

    # Draw contour in the mask
    cv2.drawContours(blank_image, [approx], -1, (255, 255, 255), -1)

    # Create a mask to select pixels inside the figure
    mask_contour = blank_image == 255

    # Calculate the intensity from the grayscale image
    # filtering out the pixels where in the blank_image their value is not 255
    intensity = np.mean(gray[mask_contour])
    print(f'{intensity=}')

    cv2.imshow("Image", blank_image)
    cv2.waitKey(0)