import cv2
import numpy as np
from matplotlib import pyplot as plt
import dicom

# Load data and create the matrix of the image
def valueHistogram(img_dicom):

    # Concentrate only between 100 and 180 value because of it's the color object
    hist = cv2.calcHist([img_dicom], [0], None, [30], [100, 130])
    # print(hist)
    sumValueHist = 0
    i = 0
    for el in hist:
        sumValueHist += el
        i += 1

    average = sumValueHist / i
    canny = [200, 200]
    median_blur = 5
    kernel_erode = np.ones((1, 1), np.uint8)
    if average [0] < 400 :
        if 200 < average[0] < 400 :
            print(
            "-------------------------------100 < average[0] < 300  ----------------------------------------------------------",
            average[0])
            median_blur = 5
            canny = [230, 230]
            kernel_erode = np.ones((1, 1), np.uint8)
        elif 150 < average[0] < 200:
            print(
            "-------------------------------100 < average[0] < 150  ----------------------------------------------------------",
            average[0])
            median_blur = 5
            canny = [180, 180]
            kernel_erode = np.ones((1, 1), np.uint8)
        elif 100 < average[0] < 150:
            print(
            "-------------------------------100 < average[0] < 150  ----------------------------------------------------------",
            average[0])
            median_blur = 9
            canny = [170, 170]
            kernel_erode = np.ones((1, 1), np.uint8)

        elif 50 < average[0] < 100 :
            print("-------------------------------50 < average[0] < 100  ----------------------------------------------------------",average[0])
            median_blur = 7
            canny = [225,225]
            kernel_erode = np.ones((1, 1), np.uint8)


        elif  40 < average[0] < 50  :
            print("-------------------------------40 < average[0] < 50  ----------------------------------------------------------",average[0])


            canny = [230, 230]
            median_blur = 7
            kernel_erode = np.ones((1, 1), np.uint8)

        elif  30 < average[0] < 40  :
            print("-------------------------------30 < average[0] < 40  ----------------------------------------------------------",average[0])

            canny = [230, 230]
            median_blur = 7
            kernel_erode = np.ones((1, 1), np.uint8)

        elif 10 < average[0] < 30 :
            print("-------------------------------10 < average[0] < 30  ----------------------------------------------------------",average[0])

            canny = [230, 230]
            median_blur = 7
            kernel_erode = np.ones((1, 1), np.uint8)


        elif 5 < average[0] < 10 :
            print("-------------------------------5 < average[0] < 10  ----------------------------------------------------------",average[0])

            canny = [90, 90]
            median_blur = 11
            kernel_erode = np.ones((1, 1), np.uint8)

        elif  average[0] < 5 :
            print("------------------------------average[0] < 5  ----------------------------------------------------------",average[0])
            median_blur = 7
            canny = [225, 225]

            kernel_erode = np.ones((1, 1), np.uint8)
        else :
            print(
            "-------------------------------100 < average[0] < 300  ----------------------------------------------------------",
            average[0])

    else :
        print("------------------------------- average[0] >300  ----------------------------------------------------------",average[0])
        median_blur = 5
        canny = [210, 210]
        kernel_erode = np.ones((1, 1), np.uint8)

    return average[0],median_blur,canny,kernel_erode









# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.fa25.fr.rothschild.S.4674027.1_00000.DCM'

# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.1a49.fr.rothschild.S.4818696.1_00000.DCM'
# file_name = '/home/camelot/Vidéos/angios/test1.DCM'

# data_dicom = dicom.read_file(file_name)
# img_dicom = np.array(data_dicom.pixel_array[10],np.uint8)
# img_dicom = img_dicom[y1_crope:y2_crope, x1_crope:x2_crope]
#
# print(valueHistogram(img_dicom))
