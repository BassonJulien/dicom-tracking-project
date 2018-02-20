import cv2
import numpy as np
import bezier
from matplotlib import pyplot as plt
from scipy.misc import comb

# Draw the trajectory in 2D
# Points must be an array of tuple each tuple contain the x and y coordinates

def draw_trajectory(points):
    img = np.zeros((1024, 1024, 3), dtype='uint8')
    for j in range(0, len(points)):
        if j != (len(points) - 1):
            pt1 = (int(points[j][0]), int(points[j][1]))
            pt2 = (int(points[j+1][0]), int(points[j+1][1]))
            cv2.line(img, pt1, pt2, (255, 0, 0), 1)

    winname = 'trajectory'
    cv2.namedWindow(winname)
    cv2.imshow(winname, img)
    cv2.waitKey(0)


def lissage(points):
    xvals, yvals = bezier_curve(points, nTimes=100)

    listePoints = list()

    for i in range(0, len(xvals)):
        listTransition = []
        listTransition.append(xvals[i])
        listTransition.append(yvals[i])
        listePoints.append(listTransition)

    draw_trajectory(listePoints)

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])


    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals
