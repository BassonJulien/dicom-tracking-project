import cv2
import numpy as np


# Draw the trajectory in 2D
# Points must be an array of tuple each tuple contain the x and y coordinates

def draw_trajectory(points):
    img = np.zeros((1024, 1024, 3), dtype='uint8')
    for j in range(0, len(points)):
        if j != (len(points) - 1):
            pt1 = points[j]
            pt2 = points[j + 1]
            cv2.line(img, pt1, pt2, (255, 0, 0), 3)

    winname = 'trajectory'
    cv2.namedWindow(winname)
    cv2.imshow(winname, img)
    cv2.waitKey()
