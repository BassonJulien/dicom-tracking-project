import numpy as np
import cv2
import sys
sys.path.append('../')
from orb_process import orbDetectionCrop, preprocess

width_window = 300
crop_constante = 75


def neuron_preprocess (img, refPoint,shift_x,shift_y):

    # make a copy of the image to not update the original
    img_dicom = img.copy()
    try:
        # try if there are multiple refpoints

        print("3  eme et + it' ",refPoint[0][0],shift_x,shift_y)
        prevPointX = refPoint[0][0]
        prevPointY = refPoint[0][1]
        print("3  eme et + it' ",refPoint[1][0])
        pointX = refPoint[1][0] + int(shift_x)
        pointY = refPoint[1][1] + int(shift_y)

        # Var for crop image, to know the decalage with the previous refPoint
        pointX1_crop = int(pointX - crop_constante)
        pointX2_crop = int(pointX + crop_constante)
        pointY1_crop = int(pointY - crop_constante)
        pointY2_crop = int(pointY + crop_constante)

        x_origin = pointX
        y_origin = pointY


    except:
        # Only one refPoint

        print("2 eme it' ",refPoint[0],shift_x,shift_y)

        # Last refPoint detected
        pointX = refPoint[0]+int(shift_x)
        pointY = refPoint[1]+int(shift_y)
        # Set the new repere origin
        x_origin = pointX
        y_origin = pointY

        # Var for crop image, to know the decalage with the previous refPoint
        pointX1_crop = int(pointX - crop_constante)
        pointX2_crop = int(pointX + crop_constante)
        pointY1_crop = int(pointY - crop_constante)
        pointY2_crop = int(pointY + crop_constante)


    # Crop image for the orb detection
    print("cropage", pointX1_crop, pointY1_crop)
    if pointY1_crop < 0:
        pointY1_crop += 50
        pointY2_crop += 50
        img_dicom = img_dicom[pointY1_crop:pointY2_crop, pointX1_crop:pointX2_crop]

    else:
        img_dicom = img_dicom[pointY1_crop:pointY2_crop, pointX1_crop:pointX2_crop]
        # cv2.imshow('image_crop', img_dicom)
        # cv2.waitKey(0)
    print("origin ", x_origin, y_origin)

    return  img_dicom,x_origin,y_origin


def detection_process (img_dicom, refPoint,i,model) :


    # Matches between template and the image
    if refPoint[0] is None:
        prepro = preprocess.preprocess(img_dicom, refPoint, i)
        matches = prepro[0]
        kp1 = prepro[1]
        img = prepro[2]
        x_cropeTot = prepro[3]
        y_cropeTot = prepro[4]
        nbr_matches = 20
        try:
            refPoint[0][0]
            point = orbDetectionCrop.verifyObject(matches[:nbr_matches], kp1, refPoint[1])
            print("main", point)
            pointX = point[0]
            pointY = point[1]
            refPoint = [pointX + x_cropeTot, pointY + y_cropeTot]
        except:
            point = orbDetectionCrop.verifyObject(matches[:nbr_matches], kp1, refPoint)
            print("main", point)
            pointX = point[0]
            pointY = point[1]
            refPoint = [pointX + x_cropeTot, pointY + y_cropeTot]

    else :
        print("furtur neuron",refPoint)
        b_found = False
        b_found_x = False
        tabRefPoint = []
        y = 0
        for shift_y in range(int(-30),int(30),5):
            i = 0
            for shift_x in range(int(-60), int(60),10):
                prepro = neuron_preprocess(img_dicom, refPoint,shift_x,shift_y)
                img = np.array([prepro[0]])
                # height, width,channels = img.shape
                prediction_classe = model._predict_byOneImage(model,img)[0]
                x_cropeTot = prepro[1]
                y_cropeTot = prepro[2]
                # print(shift_x,shift_y,i,prediction_classe,x_cropeTot,y_cropeTot)
                pointX = x_cropeTot
                pointY = y_cropeTot
                if prediction_classe == 1 :
                    tabRefPoint.append([pointX,pointY])

                print("la valeur du point : ",pointX,pointY)
        sumX = 0
        sumY = 0
        nbrPoints = len(tabRefPoint)
        for point in tabRefPoint :
            try :
                print(point)
                sumX = sumX + point[0]
                sumY = sumY + point[1]
            except :
                sumX = sumX + point[0][0]
                sumY = sumY + point[0][1]


        refPoint = [sumX/nbrPoints,sumY/nbrPoints]
        print("resultat : ",sumX,sumY,nbrPoints,refPoint)
        # if b_found is False:
        #     print("refpoint condition : ", refPoint[0], refPoint[1])
        #     refPoint = refPoint[0]

    # Draw the rectangle region
    cv2.rectangle(img_dicom, (int(refPoint[0] - 50 ), int(refPoint[1]  -50 )), (int(refPoint[0] + 50.00), int( 50 + refPoint[1])),
                  (255, 0, 0), 2)
    cv2.circle(img_dicom, (int(pointX  + x_cropeTot),int(pointY  + y_cropeTot)), 5, (255, 255, 0), -1)  # draw center


    cv2.namedWindow('original image ', cv2.WND_PROP_FULLSCREEN)
    cv2.imshow('original image', img_dicom)


    return img_dicom,refPoint
