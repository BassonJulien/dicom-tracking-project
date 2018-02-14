import numpy as np
import cv2
from matplotlib import pyplot as plt
import dicom

# Data header of the dicom file
data_dicom = dicom.read_file("./dicom/test1.DCM")
# Image of a catheter model
template = cv2.imread("/home/camelot/workspace/dicom-tracking-project/dicom/templateCANNYEDGES.png")
template2 = cv2.imread("/home/camelot/workspace/dicom-tracking-project/dicom/templateCANNYEDGES2.png")

img_dicom = np.array(data_dicom.pixel_array[10],np.uint8)
img_dicom = img_dicom[260:700, 200:810]

img_blur = cv2.medianBlur(img_dicom,5)
# Detection of edges
edges = cv2.Canny(img_blur,100,200)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(edges,None)
kp2, des2 = sift.detectAndCompute(template,None)
kp3, des3 = sift.detectAndCompute(template2, None
                                  )
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
matches2 = bf.knnMatch(des1,des3, k=2)
def checkBestMatches (matches):
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    return good


good = checkBestMatches(matches)
good2 = checkBestMatches(matches2)

# print(matches[0][0].trainIdx)
def average (good):
    sumX = 0
    sumY = 0
    size = 0
    for mat in good:
        img1_idx = mat[0].queryIdx
        img2_idx = mat[0].trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        sumX += x1
        sumY += y1
        size+=1
    x=sumX/size
    y=sumY/size
    return(x,y)


x,y = average(good)
x1,y1 = average(good2)
print((x+x1)/2,(y+y1)/2)
good4 = good + good2
# print(good4)
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(edges, kp1, template, kp2, good[:30], None, flags=2)
img4 = cv2.drawMatchesKnn(edges, kp1, template2, kp3, good2[:5], None, flags=2)

plt.imshow(img4)

plt.show()
