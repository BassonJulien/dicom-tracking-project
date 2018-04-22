import dicom
import numpy as np
import cv2
from matplotlib import pyplot as plt
import histogramme





def preprocess (img, refPoint, i) :
    crop_constante = 75
    # Image of a catheter model
    template = cv2.imread("./template/templateCANNYEDGES.png")

    # make a copy of the image to not update the original
    img_dicom = img.copy()

    # ---------------First frame no refpoint---------------------------------------------------------------------------
    if refPoint[0] is  None:
        print("premiere iteration", refPoint)

        histValue = histogramme.valueHistogram(img_dicom)[0]
        print( "la valeur de l'histo est ",histValue)

        if 600 < histValue < 3100 :

            x1_crope = 200
            x2_crope = 900
            y1_crope = 260
            y2_crope = 750
        elif 3100 <histValue < 3200 :
            template = cv2.imread("/home/camelot/workspace/dicom-tracking-project/template/teplate3.png")
            x1_crope = 200
            x2_crope = 900
            y1_crope = 260
            y2_crope = 750

        elif 4810 <histValue < 4820 :
            # template = cv2.imread("/home/camelot/workspace/dicom-tracking-project/template/teplate3.png")
            x1_crope = 200
            x2_crope = 1100
            y1_crope = 260
            y2_crope = 900

        else :
            x1_crope = 200
            x2_crope = 810
            y1_crope = 200
            y2_crope = 750

        img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]
        # Origin is only the normal crop
        x_origin = x1_crope
        y_origin = y1_crope

    # -------------------2n Frame and n+1 frame and Cropage----------------------------------------------------------
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

        except Exception as e:
            print(e)
            # Only one refPoint
            print("2 eme it' ")

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
        print("cropage",pointX1_crop, pointY1_crop)
        if pointY1_crop < 0 :
            pointY1_crop += 50
            pointY2_crop += 50
            img_dicom = img_dicom[pointY1_crop:pointY2_crop, pointX1_crop:pointX2_crop]

        else :
            img_dicom = img_dicom[pointY1_crop:pointY2_crop, pointX1_crop:pointX2_crop]
        cv2.imshow('image_crop', img_dicom)
        cv2.waitKey(0)


    # ----------------------------------------------Histogram-------------------------------------------------
    # To determine the difference between the frame and video to manage parameter in segmentation functions
    numImagetemplate = i
    # cv2.imwrite('/home/camelot/workspace/dicom-tracking-project/train/templates4(%d).png' % numImagetemplate, img_dicom)
    plt.show()

    histo = histogramme.valueHistogram(img_dicom)
    MEDIAN_BLUR = histo[1]
    CANNY = histo[2]
    KERNEL_ERODE = histo[3]

    img_blur = cv2.medianBlur(img_dicom, MEDIAN_BLUR)
    if refPoint[0] is None and 400<histo[0]<600:
        img_blur = cv2.dilate(img_blur, KERNEL_ERODE, iterations=1)

    if  0<histo[0]<5:
        img_blur = cv2.dilate(img_blur, KERNEL_ERODE, iterations=1)

    edges = cv2.Canny(img_blur, CANNY[0], CANNY[1])




    cv2.imshow('edges', edges)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(edges, None)
    kp2, des2 = orb.detectAndCompute(template, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    if len(matches) < 3 :
        print('no object detected so we need to increase the ROI size', matches)
        img_dicom = img[pointY1_crop-30:pointY2_crop+30, pointX1_crop-30:pointX2_crop+30]
        img_blur = cv2.medianBlur(img_dicom, MEDIAN_BLUR)
        edges = cv2.Canny(img_blur, CANNY[0], CANNY[1])
        kp1, des1 = orb.detectAndCompute(edges, None)
        matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    print("les matches sont : ",matches)
    # Draw the matches between the template and the image
    image_matches = cv2.drawMatches(edges, kp1, template, kp2, matches[:15], None, flags=2)
    cv2.imshow('ORB', image_matches)
    cv2.imshow('blur', img_blur)
    cv2.imshow('Template', template)

    return matches, kp1, img_dicom,x_origin,y_origin
