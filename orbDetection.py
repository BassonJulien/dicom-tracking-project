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
template = cv2.imread("/template/templateCANNYEDGES.png")
template2 = cv2.imread("template/templateCANNYEDGES2.png")

# check if the object is the catheter
def verifyObject(good,kp1,refPoint):

    point = [None,None,None]
    size_object_recongnize = [0,0]
    size_good = np.shape(good)[0]
    x = point[0]
    y = point[1]

    # print("refPoint : ",refPoint)

    # Test for each matches points
    for i in range(0,size_good-2):
        if point[0] is None :
            print("rentre dans none")
            point = average(good[i:], kp1, refPoint)
            x = point[0]
            y = point[1]
        if refPoint[1] is not None :
            if x < refPoint[0] + 10 and y < refPoint[1] + 10 and x > refPoint[0] - 10 and y > refPoint[1] - 10:
                print('ok I found the catheter')
                break
        else:
            if size_object_recongnize[0] >= round(0.40 * size_good):
                print('ok I found the catheter')
                break


            print('search...',i,size_good)
            point = average(good[i:],kp1,refPoint)
            if point[2]>size_object_recongnize[0]:
                print("superieur a la precedente",size_object_recongnize[0])
                size_object_recongnize = [point[2], i]
                point = average(good[size_object_recongnize[1]:], kp1, refPoint)

            if i == size_good-4:
                print("rentre dans la condition")
                point = average(good[size_object_recongnize[1]:], kp1, refPoint)
                break
            print(point[0],point[1],point[2],size_object_recongnize[1])
    print(point,size_good, good)
    return point


def average(good,kp1,refPoint):

    sumX = 0
    sumY = 0
    size = 0
    reX,refY = 0,0
    size_good = np.shape(good)[0]
    print("test")

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
    print("finito",sumX,sumY,size)
    # Calculate average points
    x = sumX / size
    y = sumY / size

    return x, y, size

def preprocess (img, refPoint) :
    # img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]
    img_dicom = img.copy()
    print("yolo",refPoint)
    if refPoint[0] is  None:
        img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]
        x_cropeTot = x1_crope
        y_cropeTot = y1_crope


    else :
        # img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]
        # x_cropeTot = x1_crope
        # y_cropeTot = y1_crope
        img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]
        print("premiere condi")
        pointX1 = int(refPoint[0]-100)
        pointX2 = int(refPoint[0]+200)
        pointY1 = int(refPoint[1]-100)
        pointY2 = int(refPoint[1]+200)
        img_dicom = img_dicom[pointY1:pointY2, pointX1:pointX2]
        x_cropeTot = x1_crope
        y_cropeTot = y1_crope
        cv2.imshow('image_dilate', img_dicom)
        cv2.waitKey(0)

    # elif refPoint[0] < 100 and refPoint[1] < 100:
    #     img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]
    #     print("premiere condi")
    #     # pointX1 = int(refPoint[0]-70)
    #     # pointX2 = int(refPoint[0]+70)
    #     # pointY1 = int(refPoint[1]-70)
    #     # pointY2 = int(refPoint[1]+70)
    #     # img_dicom = img_dicom[pointY1:pointY2, pointX1:pointX2]
    #     # cv2.imshow('image_dilate', img_dicom)
    #     # cv2.waitKey(0)
    #
    # elif refPoint[0] > 100 and refPoint[1] > 100:
    #     img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]
    #     print("2eme condi")
    #
    #     pointX1 = int(refPoint[0] - 100)
    #     pointX2 = int(refPoint[0] + 100)
    #     pointY1 = int(refPoint[1] - 100)
    #     pointY2 = int(refPoint[1] + 100)
    #     img_dicom = img_dicom[pointY1:pointY2, pointX1:pointX2]
    #     # cv2.imshow('image_dilate', img_dicom)
    #     # cv2.waitKey(0)
    #
    # elif refPoint[0] > 200 and refPoint[1] > 200:
    #     img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]
    #     print("3eme condi")
    #
    #     pointX1 = int(refPoint[0] - 200)
    #     pointX2 = int(refPoint[0] + 200)
    #     pointY1 = int(refPoint[1] - 200)
    #     pointY2 = int(refPoint[1] + 200)
    #     img_dicom = img_dicom[pointY1:pointY2, pointX1:pointX2]
    #     # print(img_dicom)
    # else :
    #     print("else condi")
    #     img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]
    #     print("premiere condi")
    #     pointX1 = int(refPoint[0]-30)
    #     pointX2 = int(refPoint[0]+30)
    #     pointY1 = int(refPoint[1]-30)
    #     pointY2 = int(refPoint[1]+30)
    #     img_dicom = img_dicom[pointY1:pointY2, pointX1:pointX2]
    #     cv2.imshow('image_dilate', img_dicom)
    #     cv2.waitKey(0)

    # Filtrage to decrease noise
    img_blur = cv2.medianBlur(img_dicom, 5)

    if refPoint[0] is None:
        # Detection of edges
        edges = cv2.Canny(img_blur, 200, 200)
    else :
        edges = cv2.Canny(img_blur, 100, 200)
        # cv2.imshow('image_dilate', edges)
        # cv2.waitKey(0)

    # Erosion
    # kernel_erode = np.ones((1,1),np.uint8)
    # image_erode = cv2.erode(edges, kernel_erode, iterations = 1)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(edges, None)
    kp2, des2 = orb.detectAndCompute(template, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    print("les matches sont : ",matches)
    # Draw the matches between the template and the image
    # image_matches = cv2.drawMatches(edges, kp1, template, kp2, matches[:15], None, flags=2)

    return matches, kp1, img_dicom,x_cropeTot, y_cropeTot


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
    prepro =  preprocess(img_dicom, refPoint)
    matches = prepro[0]
    kp1 = prepro[1]
    img = prepro[2]
    x_cropeTot = prepro[3]
    y_cropeTot = prepro[4]

    # Matches between template and the image

    point = verifyObject(matches[:15], kp1, refPoint)
    pointX = point[0]
    pointY = point[1]
    refPoint = [pointX  ,pointY ]
    # refPoint = [pointX +x_cropeTot ,pointY + y_cropeTot]
    print("refPoint : ",refPoint)
    print([pointX +x_cropeTot ,pointY + y_cropeTot])
    print("main", point)
    # Draw the rectangle region
    cv2.rectangle(img_dicom, (int(pointX - 50 + x_cropeTot), int(pointY - 50 + y_cropeTot)), (int(pointX + 50.00 + x_cropeTot), int(pointY + 50.00 + y_cropeTot)),
                  (255, 0, 0), 2)



    # plt.imshow(image_matches), plt.show()
    # cv2.imshow('edges', img_dicom)

    return img_dicom,refPoint
