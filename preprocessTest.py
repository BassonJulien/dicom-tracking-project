import dicom
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Set threshold and maxValue
thresh = 115
maxValue = 117
template = cv2.imread("/home/camelot/workspace/dicom-tracking-project/template/NOISY_TEMPLATE2.png")

# Data header of the dicom file
file_name = '/home/camelot/Vidéos/angios/ARX1.rot.fa25.fr.rothschild.S.4674027.1_00000.DCM'

data_dicom = dicom.read_file(file_name)
# Image of a catheter model


img_dicom = np.array(data_dicom.pixel_array[0],np.uint8)
# img_dicom = img_dicom[260:700, 200:810]

img_dicom = cv2.medianBlur(img_dicom,7)


# ret, img_dicom_thresh = cv2.threshold(img_dicom, thresh, maxValue, cv2.THRESH_BINARY_INV)

# Histogram
# plt.hist(img_dicom.ravel(),256,[0,256]); plt.show()

# MedianBlur permit to reduce/blur noise
# img_blur = cv2.GaussianBlur(img_blur,(5,5),0)
# img_laplacian = cv2.Laplacian(img_dicom_thresh,cv2.CV_64F)


# Dilatation
kernel_dilate = np.ones((1,1),np.uint8)
image_erode = cv2.dilate(img_dicom, kernel_dilate, iterations = 1)

# # Erosion
# kernel_erode = np.ones((1,1),np.uint8)
# image_erode = cv2.erode(img_dicom, kernel_erode, iterations = 1)


# Detection of edges
edges = cv2.Canny(image_erode,130,130)

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(image_erode, None)
kp2, des2 = orb.detectAndCompute(template, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
# matches = bf.match(des1, des2)

# Sort them in the order of their distance.
# matches = sorted(matches, key=lambda x: x.distance)

# img3 = cv2.drawMatches(image_erode, kp1, template, kp2, matches[:5], None, flags=2)

# plt.imshow(img3), plt.show()
# plt.imshow(image_dilate)s
# plt.subplot(121),plt.imshow(img_dicom,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# # plt.subplot(122),plt.imshow(img_dicom_thresh,cmap = 'gray')
# # plt.title('img_dicom_thresh'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(image_dilate,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()


cv2.imshow('image_dilate', edges)

cv2.waitKey(0)
