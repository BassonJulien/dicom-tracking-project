#!/usr/bin/python
# -*- coding: utf-8 -*-

import dicom
import numpy as np
import cv2
from matplotlib import pyplot as plt


# Constante
x1_crope = 200
x2_crope = 810
y1_crope = 260
y2_crope = 700

file_name = '/home/camelot/Vidéos/angios/test1.DCM'

# Data header of the dicom file
data_dicom = dicom.read_file(file_name)

# Image of a catheter model
template = cv2.imread("/home/camelot/workspace/dicom-tracking-project/template/templateCANNYEDGES.png")

# Create matrix image and crope it
img_dicom = np.array(data_dicom.pixel_array[5],np.uint8)
img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]

# Filtrage to decrease noise
img_blur = cv2.medianBlur(img_dicom,5)


# Detection of edges
edges = cv2.Canny(img_blur,100,200)

# Erosion
# kernel_erode = np.ones((1,1),np.uint8)
# image_erode = cv2.erode(edges, kernel_erode, iterations = 1)

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(edges, None)
kp2, des2 = orb.detectAndCompute(template, None)
kp2, des2 = orb.detectAndCompute(template, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)
# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)


# check if the object is the catheter
def verifyObject(good):

    point = average(good)
    size_object_recongnize = point[2]
    size_good = np.shape(good)[0]

    # If the object tracked have more than the half of matches points, it's catheter
    if size_object_recongnize >= round(0.70 * size_good):
        print('ok I found the catheter')

    # Test for each matches points
    else:
        for i in range(0,size_good):
            if size_object_recongnize >= round(0.50 * size_good):
                print('ok I found the catheter')
                break
            else:
                print('search...')
                point = average(good[i:])
                size_object_recongnize = point[2]
                print(point[0],point[1],point[2])
    return point


def average(good):

    sumX = 0
    sumY = 0
    size = 0
    reX,refY = 0,0
    size_good = np.shape(good)[0]
    for i in range(0,size_good):

        # Get the matching keypoints for each of the images
        img1_idx = good[i].queryIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt

        # To init the region with "normally" the best matches point
        if size==0 :
            print("init")
            refY= y1
            refX= x1
            sumX += x1
            sumY += y1
            size+=1

        # Work only in the region of the reference point(here the first point)
        if x1<refX+50 and y1<refY+50 and x1 > refX-50 and y1 > refY-50:
            print("test",refX,x1,y1)
            sumX += x1
            sumY += y1
            size += 1

    # Calculate average points
    x = sumX/size
    y = sumY/size

    return x, y, size


point = verifyObject(matches[:15])
pointX = point[0]
pointY = point[1]

# Draw the matches between the template and the image
image_matches = cv2.drawMatches(edges, kp1, template, kp2, matches[:15], None, flags=2)

# Draw the rectangle region
cv2.rectangle(image_matches, (int(pointX-100), int(pointY-100)), (int(pointX+100.00), int(pointY+100.00)), (255,0,0), 2)

plt.imshow(image_matches), plt.show()
cv2.imshow('edges', img_dicom)
cv2.waitKey(0)
