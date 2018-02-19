import cv2
import numpy as np
import dicom

# Constante
x1_crope = 200
x2_crope = 810
y1_crope = 260
y2_crope = 750
def findCircle (img):
    file_name = '/home/camelot/Vidéos/angios/test1.DCM'

    data_dicom = dicom.read_file(file_name)
    img = np.array(data_dicom.pixel_array[201], np.uint8)
    img = img[y1_crope:y2_crope, x1_crope:x2_crope]
    # img = cv2.medianBlur(img, 5)
    ret, img = cv2.threshold(img, 105, 130, cv2.THRESH_BINARY_INV)

    # img = cv2.Canny(img, 100, 200)
    kernel_dilate = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel_dilate, iterations=1)
    output = img.copy()

    # detect circles in the image
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300)

    # ensure at least some circles were found
    if circles is not None:
        biggerCircle = circles[0][0]
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            if biggerCircle[2] < r:
                biggerCircle = [x, y, r]
                cv2.circle(output, (x, y), r, (0, 255, 0), -1)

            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle

        cv2.circle(output, (biggerCircle[0], biggerCircle[1]), biggerCircle[2], (0, 255, 0), -1)
        # cv2.rectangle(output, (biggerCircle[0] - 5, biggerCircle[1] - 5), (biggerCircle[0] + 5, biggerCircle[1] + 5),
        #               (0, 128, 255), -1)
        # # show the output image
        # cv2.imshow("output", output)
        # cv2.waitKey(0)
        cv2.imshow('edges', output)
        cv2.waitKey(0)

        return biggerCircle

