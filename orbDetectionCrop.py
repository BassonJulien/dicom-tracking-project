import dicom
import numpy as np
import cv2
from matplotlib import pyplot as plt
import preprocess


# check if the object is the catheter
def verifyObject(good,kp1,refPoint):

    point = [None,None,None]
    size_object_recongnize = [0,0]
    size_good = np.shape(good)[0]
    x = point[0]
    y = point[1]

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



def main_orb_detection (img_dicom, refPoint,i) :

    prepro =  preprocess.preprocess(img_dicom, refPoint, i)
    matches = prepro[0]
    kp1 = prepro[1]
    img = prepro[2]
    x_cropeTot = prepro[3]
    y_cropeTot = prepro[4]
    nbr_matches = 20
    # Matches between template and the image
    try :
        refPoint[0][0]
        point = verifyObject(matches[:nbr_matches], kp1, refPoint[1])
        print("main", point)
        pointX = point[0]
        pointY = point[1]
        refPoint = [pointX + x_cropeTot, pointY + y_cropeTot]
    except:
        point = verifyObject(matches[:nbr_matches], kp1, refPoint)
        print("main", point)
        pointX = point[0]
        pointY = point[1]
        refPoint = [pointX + x_cropeTot, pointY+y_cropeTot]


    print("refPoint : ",refPoint)

    # Draw the rectangle region
    cv2.rectangle(img_dicom, (int(pointX - 50 + x_cropeTot), int(pointY - 50 + y_cropeTot)), (int(pointX + 50.00 + x_cropeTot), int(pointY + 50.00 + y_cropeTot)),
                  (255, 0, 0), 2)

    cv2.namedWindow('original image ', cv2.WND_PROP_FULLSCREEN)
    cv2.imshow('original image', img_dicom)


    return img_dicom,refPoint

# cv2.waitKey(0)
