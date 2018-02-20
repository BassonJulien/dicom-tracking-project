import dicom
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Constante
x1_crope = 200
x2_crope = 810
y1_crope = 260
y2_crope = 750
crop_constante = 100
# Image of a catheter model
template = cv2.imread("/home/camelot/workspace/dicom-tracking-project/template/templateCANNYEDGES.png")
template2 = cv2.imread("/home/camelot/workspace/dicom-tracking-project/template/templateCANNYEDGES2.png")

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
                print("superieur à la precedente",size_object_recongnize[0])
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

    # make a copy of the image to not update the original
    img_dicom = img.copy()
    cv2.namedWindow('grande', cv2.WINDOW_NORMAL)
    cv2.imshow('grande', img_dicom)
    # cv2.resizeWindow('grande', 600, 600)

    # First frame no refpoint
    if refPoint[0] is  None:
        print("premiere iteration", refPoint)
        img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]
        # Origin is only the normal crop
        x_origin = x1_crope
        y_origin = y1_crope

    # Catheter detected
    else :
        print("Cree les sous reperes",refPoint)
        # Normal crope

        try :
            # try if there are multiple refpoints

            print("3  eme et + it' ")
            prevPointX =refPoint[0][0]
            prevPointY = refPoint[0][1]
            pointX =refPoint[1][0]
            pointY = refPoint[1][1]

            # Var for crop image, to know the decalage with the previous refPoint
            pointX1_crop = int(pointX - crop_constante)
            pointX2_crop = int(pointX + crop_constante)
            pointY1_crop = int(pointY - crop_constante)
            pointY2_crop = int(pointY + crop_constante)


            x_origin = int(pointX1_crop)
            y_origin = int(pointY1_crop)


        except:
            # Only one refPoint

            print("2 eme it' ")
            # prevPointX = 0
            # prevPointY = 0
            # Last refPoint detected
            pointX = refPoint[0]
            pointY = refPoint[1]
            # Set the new repere origin
            x_origin = pointX - crop_constante
            y_origin = pointY - crop_constante

            # Var for crop image, to know the decalage with the previous refPoint
            pointX1_crop = int(pointX - crop_constante)
            pointX2_crop = int(pointX + crop_constante)
            pointY1_crop = int(pointY - crop_constante)
            pointY2_crop = int(pointY + crop_constante)

        # Crop image for the orb detection
        print("cropage",pointY1_crop,pointX1_crop)
        img_dicom = img_dicom[pointY1_crop:pointY2_crop, pointX1_crop:pointX2_crop]
        cv2.imshow('image_dilate', img_dicom)
        cv2.waitKey(0)
        # To know the coordinates repere origin

        # cv2.imshow('image_dilate', img_dicom)
        # cv2.waitKey(0)

    # Filtrage to decrease noise
    img_blur = cv2.medianBlur(img_dicom, 5)

    if refPoint[0] is None:
        # Detection of edges
        edges = cv2.Canny(img_blur, 200, 200)
    else :
        edges = cv2.Canny(img_blur, 100, 200)
        # cv2.imshow('image_dilate', edges)
        # cv2.waitKey(0)

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

    return matches, kp1, img_dicom,x_origin,y_origin


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
    try :
        refPoint[0][0]
        print('-------------------------------------------------------------------------------------quequette')
        point = verifyObject(matches[:15], kp1, refPoint[1])
        print("main", point)
        pointX = point[0]
        pointY = point[1]
        refPoint = [pointX + x_cropeTot, pointY + y_cropeTot]
    except:
        print('-------------------------------------------------------------------------------------')
        point = verifyObject(matches[:15], kp1, refPoint)
        print("main", point)
        pointX = point[0]
        pointY = point[1]
        refPoint = [pointX + x_cropeTot, pointY+y_cropeTot]


    print("refPoint : ",refPoint)
    # print([pointX +x_cropeTot ,pointY + y_cropeTot])
    # print("main", point)

    # Draw the rectangle region
    cv2.rectangle(img_dicom, (int(pointX - 50 + x_cropeTot), int(pointY - 50 + y_cropeTot)), (int(pointX + 50.00 + x_cropeTot), int(pointY + 50.00 + y_cropeTot)),
                  (255, 0, 0), 2)



    # plt.imshow(image_matches), plt.show()
    # cv2.imshow('edges', img_dicom)

    return img_dicom,refPoint

# cv2.waitKey(0)
