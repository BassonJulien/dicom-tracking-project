import dicom
import numpy as np
import cv2
from matplotlib import pyplot as plt
import histogramme
import orbDetectionCrop
import preprocess

width_window = 300
def neuron_preprocess (img, refPoint,shift_x,shift_y):
    crop_constante = 75

    # make a copy of the image to not update the original
    img_dicom = img.copy()
    try:
        # try if there are multiple refpoints

        print("3  eme et + it' ",refPoint[0][0])
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

        x_origin = int(pointX1_crop)
        y_origin = int(pointY1_crop)


    except:
        # Only one refPoint

        print("2 eme it' ",refPoint[0])

        # Last refPoint detected
        pointX = refPoint[0]+int(shift_x)
        pointY = refPoint[1]+int(shift_y)
        # Set the new repere origin
        x_origin = pointX - crop_constante
        y_origin = pointY - crop_constante

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
        try :
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
            refPoint = [pointX + x_cropeTot, pointY+y_cropeTot]

    else :
        print("furtur neuron")
        img = None
        x_cropeTot = None
        y_cropeTot = None
        b_found = False
        for shift_y in range(int(-50),int(50),10):
            i = 0
            for shift_x in range(int(-50), int(50),10):
                prepro = neuron_preprocess(img_dicom, refPoint,shift_x,shift_y)
                img = np.array([prepro[0]])
                prediction_classe = model._predict_byOneImage(model,img)[0]
                print(shift_x,shift_y,i,prediction_classe)
                x_cropeTot = prepro[1]
                y_cropeTot = prepro[2]
                pointX = x_cropeTot + 75
                pointY = y_cropeTot + 75
                print()
                if prediction_classe == 1 :
                    refPoint = [75 + x_cropeTot, 75 + y_cropeTot]
                    b_found = True
                    break
            if b_found is True :
                break

    # Draw the rectangle region
    cv2.rectangle(img_dicom, (int(75 - 50 + x_cropeTot), int(150  -50 + y_cropeTot)), (int(75+ 50.00 + x_cropeTot), int(150+ 50.00 + y_cropeTot)),
                  (255, 0, 0), 2)

    # cv2.namedWindow('original image ', cv2.WND_PROP_FULLSCREEN)
    # cv2.imshow('original image', img_dicom)


    return img_dicom,refPoint
