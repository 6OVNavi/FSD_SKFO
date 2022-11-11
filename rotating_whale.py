import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img = cv2.imread('whale_train/93/DJI_0085/DJI_0085_1913.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
epsilon = 0.001 * cv2.arcLength(cnt, True)
while True:
   epsilon += epsilon
   approx = cv2.approxPolyDP(cnt, epsilon,True)
   if len(approx) <= 4:
       break

thresh[:, :] = 0
thresh = cv2.fillPoly(thresh, [approx], 255)

def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_angle(p1, p2, p3):
    """p2 = center"""
    d1 = dist(p1, p2)
    d2 = dist(p2, p3)
    angle = math.acos(min(d1, d2) / max(d1, d2))
    return math.degrees(angle)

left = list(approx[approx[:, :, 0].argmin()][0])
right = list(approx[approx[:, :, 0].argmax()][0])
other = np.array([elem[0] for elem in approx.tolist() if elem[0] not in [left, right]]).astype(int)
if len(approx) == 3:
    if dist(left, other[0]) >= dist(right, other[0]):
        start = left
        end = right
    else:
        start = right
        end = left
elif len(approx) == 4:
    if get_angle(other[0], left, other[1]) >= get_angle(other[0], right, other[1]):
        start = left
        end = right
    else:
        start = right
        end = left


contoured_img = cv2.drawContours(np.zeros(img.shape), contours, -1, 255, 5)
# blue
cv2.ellipse(contoured_img, start, (15, 15), 0, 0, 360, 70, -1)
# green
cv2.ellipse(contoured_img, end, (15, 15), 0, 0, 360, 200, -1)

# blue = start; green = end
cv2.imshow('f',contoured_img)
cv2.waitKey(0)

print(start,end)

import math

angle=math.degrees(math.atan((start[1]-end[1])/(start[0]-end[0])))

print(angle)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

rimg=rotate_image(contoured_img,angle) #

cv2.imshow('f',rimg)
cv2.waitKey(0)
