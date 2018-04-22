from draw_trajectory import plotPoints
import matplotlib.pyplot as plt


def draw2DTrajectorySmooth(list1):
    fig = plt.figure()
    ax = plt.axes()

    xvals1, yvals1 = plotPoints.bezier_curve(list1, 500)

    listTab1X = list()
    listTab1Y = list()

    for i in range(0, len(list1)):
        listTab1X.append(list1[i][0])
        listTab1Y.append(list1[i][1])

    xline = xvals1

    yline = yvals1

    ax.plot(xline, yline, 'black')

    ax.scatter(listTab1X, listTab1Y, c='green')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.gca().invert_yaxis()
    plt.show(fig)