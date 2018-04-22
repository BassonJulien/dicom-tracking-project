import dicom
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Set threshold and maxValue
thresh = 230
maxValue = 235
# Constante
x1_crope = 200
x2_crope = 900
y1_crope = 260
y2_crope = 750
template = cv2.imread("./template/templateCANNYEDGES.png")

# Data header of the dicom file
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.fc74.fr.rothschild.S.4925457.1_00000.DCM'
file_name = '/home/camelot/Vidéos/angios/test1.DCM'

data_dicom = dicom.read_file(file_name)
# Image of a catheter model
print(data_dicom)

img_dicom = np.array(data_dicom.pixel_array[100],np.uint8)
img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]

# img_dicom = img_dicom[260:700, 200:810]
# img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]

img_dicom_m = cv2.medianBlur(img_dicom,11)
# img_dicom_m = cv2.GaussianBlur(img_dicom,(7,7),0)
#
# # ret, img_dicom = cv2.threshold(img_dicom, thresh, maxValue, cv2.THRESH_BINARY_INV)
#
# # Histogram
# # plt.hist(img_dicom.ravel(),256,[0,256]); plt.show()
#
# # MedianBlur permit to reduce/blur noise
# # img_blur = cv2.GaussianBlur(img_blur,(5,5),0)
# # img_laplacian = cv2.Laplacian(img_dicom_thresh,cv2.CV_64F)
#
#
# # Dilatation
# kernel_dilate = np.ones((3,3),np.uint8)
# image_erode = cv2.erode(img_dicom, kernel_dilate, iterations = 1)
# image_erode = cv2.GaussianBlur(image_erode, (5, 5), 0) # Remove noise
# # # Erosion
# # kernel_erode = np.ones((1,1),np.uint8)
# # image_erode = cv2.erode(img_dicom, kernel_erode, iterations = 1)
#
#
# Detection of edges
edges = cv2.Canny(img_dicom_m,60,60)


# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(edges, None)
kp2, des2 = orb.detectAndCompute(template, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(edges, kp1, template, kp2, matches[:20], None, flags=2)
#
# # plt.imshow(img3), plt.show()
# # plt.imshow(edges)
# # # plt.subplot(121),plt.imshow(img_dicom,cmap = 'gray')
# # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# # # plt.subplot(122),plt.imshow(img_dicom_thresh,cmap = 'gray')
# # # plt.title('img_dicom_thresh'), plt.xticks([]), plt.yticks([])
# # plt.subplot(122),plt.imshow(image_dilate,cmap = 'gray')
# # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# # plt.show()
# def nothing(x):
#     pass
# cv2.imshow('image', image_erode)
# cv2.imshow('canny_edge', edges)
# cv2.createTrackbar('min_value', 'canny_edge', 0, 500, nothing)
# cv2.createTrackbar('max_value', 'canny_edge', 0, 500, nothing)
#
# while (1):
#     cv2.imshow('image', image_erode)
#     cv2.imshow('canny_edge', edges)
#
#     min_value = cv2.getTrackbarPos('min_value', 'canny_edge')
#     max_value = cv2.getTrackbarPos('max_value', 'canny_edge')
#
#     edges = cv2.Canny(image_erode, min_value, max_value)
#
#     k = cv2.waitKey(37)
#     if k == 27:
#         break

# cv2.imshow('détection de contour', edges)
# cv2.imshow('image_dilate', img_dicom_m)
# Initialize the plot figure
fig=plt.figure(figsize=(1, 2))
fig.add_subplot(1, 2,1)
plt.title("Original image")
plt.imshow(img_dicom, cmap='gray')
fig.add_subplot(1, 2,2)
plt.title("Orb with 20 matches points")
plt.imshow(img3, cmap='gray')
plt.show()

cv2.waitKey(0)
