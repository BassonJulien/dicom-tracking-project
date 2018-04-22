import dicom
import numpy as np
import cv2
# machine learning classification

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn import neighbors

from matplotlib import pyplot as plt


# Data header of the dicom file
data_dicom = dicom.read_file("./dicom/test1.DCM")

# Image of a catheter model
template = cv2.imread("/home/camelot/workspace/dicom-tracking-project/dicom/templateCANNYEDGES.png",0)
template2 = cv2.imread("/home/camelot/workspace/dicom-tracking-project/dicom/templateCANNYEDGES.png")
# Initialize lists
templateArray = [template,template2]
kp_train = []
des_train = []

img_dicom = np.array(data_dicom.pixel_array[30],np.uint8)
img_blur = cv2.medianBlur(img_dicom,5)
image,contours,hierachy = cv2.findContours(template,1,2)
# contours,hierarchy = cv2.findContours(template,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
M = cv2.moments(cnt)
print (M)
# cx = int(M['m10']/M['m00'])
# cy = int(M['m01']/M['m00'])
# hull = cv2.convexHull(cnt)

area = cv2.contourArea(cnt)
# perimeter = cv2.arcLength(cnt,True)
# epsilon = 0.1*cv2.arcLength(cnt,True)
# approx = cv2.approxPolyDP(cnt,epsilon,True)
# # hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]
# plt.plot(approx)
# plt.show()
# Detection of edges
edges = cv2.Canny(img_blur,100,200)

# Initiate SIFT detector
sift =  cv2.xfeatures2d.SIFT_create()

# Initiate ORB detector
orb = cv2.ORB_create()

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

# find the keypoints and descriptors with ORB
kp_query, des_query = sift.detectAndCompute(edges, None)
kp_query1, des_query1 = sift.detectAndCompute(template2, None)


# cv2.imshow('edges', edges)
# cv2.waitKey(0)
# BFMatcher with default params
bf = cv2.BFMatcher()

for el in templateArray:
    kp, des = sift.detectAndCompute(el, None)
    # print(des)
    for keyPoints in kp:
        kp_train.append(keyPoints)
    for desc in des:
        des_train.append([desc])
    # print(kp_train)

des_train = np.array(des_train,np.float32)
matches = bf.knnMatch(des_query, des_query1, k=2)

# for des in des_train:
#     print(des)
#     matches = bf.knnMatch(des_query, des, k=2)

# print(matches)
# print(matches)
# Apply ratio test
good = []
for i in range(0,len(matches)-2,2):
    # print(matches[i][0])
    if matches[i][0].distance < 0.75*matches[i+1][0].distance:
        good.append(matches[i])

# print(good)

# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(edges,kp_query,template2,kp_query1,good[:10],None,flags=2)
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,none,flags=2)
#
plt.imshow(img3),plt.show()
