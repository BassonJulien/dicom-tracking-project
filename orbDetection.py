import dicom
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Constante
x1_crope = 200
x2_crope = 810
y1_crope = 260
y2_crope = 750

# Image of a catheter model
template = cv2.imread("/home/camelot/workspace/dicom-tracking-project/template/templateCANNYEDGES.png")
template2 = cv2.imread("/home/camelot/workspace/dicom-tracking-project/template/templateCANNYEDGES2.png")

# check if the object is the catheter
def verifyObject(good,kp1,refPoint):

    point = average(good,kp1,refPoint)
    size_object_recongnize = point[2]
    size_good = np.shape(good)[0]
    x = point[0]
    y = point[1]

    print("refPoint : ",refPoint)

    # If the object tracked have more than the half of matches points, it's catheter
    if size_object_recongnize >= round(0.70 * size_good):
        print('ok I found the catheter')

    # Test for each matches points
    else:
        for i in range(0,size_good):
            if refPoint[1] is not None :
                if x < refPoint[0] + 10 and y < refPoint[1] + 10 and x > refPoint[0] - 10 and y > refPoint[1] - 10:
                    print('ok I found the catheter')
                    break
            else:
                print('search...')
                point = average(good[i:],kp1,refPoint)
                size_object_recongnize = point[2]
                print(point[0],point[1],point[2])
    return point


def average(good,kp1,refPoint):

    sumX = 0
    sumY = 0
    size = 0
    reX,refY = 0,0
    size_good = np.shape(good)[0]
    for i in range(0, size_good):

        # Get the matching keypoints for each of the images
        img1_idx = good[i].queryIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt

        # To init the region with "normally" the best matches point
        if size == 0:
            print("init")
            refY = y1
            refX = x1
            sumX += x1
            sumY += y1
            size += 1
        if refPoint[0] is not None :
            # Work only in the region of the reference point(here the first point)
            if  refX < refPoint[0] + 10 and refY < refPoint[1] + 10 and refX > refPoint[0] - 10 and refY > refPoint[1] - 10:
                if x1 < refX + 50 and y1 < refY + 50 and x1 > refX - 50 and y1 > refY - 50:
                    print("test rentre dans la condition", refX, x1, y1)
                    sumX += x1
                    sumY += y1
                    size += 1
            else :
                print("rentre pas")
                refY = refPoint[1]
                refX = refPoint[0]
                # sumX += refPoint[0]
                # sumY += refPoint[1]
                # size += 1

        else :
            if x1 < refX + 50 and y1 < refY + 50 and x1 > refX - 50 and y1 > refY - 50:
                print("test", refX, x1, y1)
                sumX += x1
                sumY += y1
                size += 1


    # Calculate average points
    x = sumX / size
    y = sumY / size

    return x, y, size

def preprocess (img_dicom) :

    img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]

    # Filtrage to decrease noise
    img_blur = cv2.medianBlur(img_dicom, 5)

    # Detection of edges
    edges = cv2.Canny(img_blur, 100, 200)

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
    # Draw the matches between the template and the image
    # image_matches = cv2.drawMatches(edges, kp1, template, kp2, matches[:15], None, flags=2)

    return matches, kp1, img_dicom


def main_orb_detection (img_dicom, refPoint) :

    # if circle is  None:
    #     # detect circles in the image
    #     circle = houghTransform.findCircle(img_dicom)
    #     cv2.circle(img_dicom, (circle[0], circle[1]), circle[2], (0, 0, 0), -1)
    #     cv2.imshow('edges', img_dicom)
    #
    #
    # else :
    #     cv2.circle(img_dicom, (circle[0], circle[1]), circle[2], (0, 0, 0), -1)

    matches = preprocess(img_dicom)[0]
    img = preprocess(img_dicom)[2]
    kp1 = preprocess(img_dicom)[1]


    # Matches between template and the image

    point = verifyObject(matches[:15], kp1, refPoint)
    pointX = point[0]
    pointY = point[1]
    refPoint = [pointX,pointY]

    # Draw the rectangle region
    cv2.rectangle(img, (int(pointX - 50), int(pointY - 50)), (int(pointX + 50.00), int(pointY + 50.00)),
                  (255, 0, 0), 2)



    # plt.imshow(image_matches), plt.show()
    # cv2.imshow('edges', img_dicom)

    return img,refPoint

