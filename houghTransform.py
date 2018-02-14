import cv2
import numpy as np
import dicom

data_dicom = dicom.read_file("./dicom/test1.DCM")
img_dicom = np.array(data_dicom.pixel_array[201],np.uint8)
img_dicom = img_dicom[260:700, 200:810]
# cimg = cv2.cvtColor(img_dicom,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img_dicom,cv2.HOUGH_GRADIENT,1,5,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img_dicom,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img_dicom,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',img_dicom)
cv2.waitKey(0)
cv2.destroyAllWindows()

