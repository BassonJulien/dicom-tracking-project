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
    if average [0] < 600 :

        if 400 < average[0] < 600 :

            print("------------------------------- 400 < average[0] >600  ----------------------------------------------------------",average[0])
            median_blur = 15
            canny = [30, 70]
            kernel_erode = np.ones((6, 6), np.uint8)

        elif 200 < average[0] < 400 :
            print(
                "-------------------------------200 < average[0] < 400  ----------------------------------------------------------",
                average[0])
            median_blur = 5
            canny = [230, 230]
            kernel_erode = np.ones((1, 1), np.uint8)

        elif 150 < average[0] < 200:
            print(
                "-------------------------------150 < average[0] < 200  ----------------------------------------------------------",
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

            canny = [0, 100]
            median_blur = 17
            kernel_erode = np.ones((1, 1), np.uint8)


        elif 10 < average[0] < 30 :
            print("-------------------------------10 < average[0] < 30  ----------------------------------------------------------",average[0])

            canny = [0, 100]
            median_blur = 17
            kernel_erode = np.ones((1, 1), np.uint8)


        elif 5 < average[0] < 10 :
            print("-------------------------------5 < average[0] < 10  ----------------------------------------------------------",average[0])

            canny = [50, 90]
            median_blur = 11
            kernel_erode = np.ones((1, 1), np.uint8)

        elif  3 < average[0] < 5 :
            print("------------------------------3 < average[0] < 5  ----------------------------------------------------------",average[0])
            median_blur = 7
            canny = [225, 225]

            kernel_erode = np.ones((1, 1), np.uint8)

        elif 0 < average[0] < 3 :
            print("------------------------------0 < average[0] < 3  ----------------------------------------------------------",average[0])
            median_blur = 7
            canny = [90, 90]

            kernel_erode = np.ones((1, 1), np.uint8)

    elif 1800 < average[0] < 1900 :
        print(
            "------------------------------- 1800 < average[0] >1900  ----------------------------------------------------------",
            average[0])
        median_blur = 21
        canny = [30, 100]
        kernel_erode = np.ones((1, 1), np.uint8)

    else:
        print(
            "------------------------------- average[0] > 600  ----------------------------------------------------------",
            average[0])
        median_blur = 5
        canny = [210, 210]
        kernel_erode = np.ones((1, 1), np.uint8)



    return average[0],median_blur,canny,kernel_erode




