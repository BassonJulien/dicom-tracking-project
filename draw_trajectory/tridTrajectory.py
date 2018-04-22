from draw_trajectory import plotPoints
import matplotlib.pyplot as plt


def draw3DTrajectory(xList1, xList2, yList1, yList2):
    fig1 = plt.figure()
    ax1 = plt.axes(projection='3d')
    yList = list()

    for i in range(0, len(yList1)):
        y = (yList1[i] + yList2[i])/2
        yList.append(y)

    xline = xList1
    yline = yList1
    zline = xList2

    ax1.plot3D(xline, yline, zline, 'black')

    zdata = [xList2[0], xList2[1], xList2[3]]
    xdata = [xList1[0], xList1[1], xList1[2]]
    ydata = [yList1[0], yList1[1], yList1[2]]

    ax1.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')

    plt.show(fig1)


def draw3DTrajectorySmooth(list1, list2):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xvals1, yvals1 = plotPoints.bezier_curve(list1, 500)
    xvals2, yvals2 = plotPoints.bezier_curve(list2, 500)

    listTab1X = list()
    listTab2X = list()
    listTab1Y = list()
    listTab2Y = list()

    for i in range(0, len(list1)):
        listTab1X.append(list1[i][0])
        listTab2X.append(list2[i][0])
        listTab1Y.append(list1[i][1])
        listTab2Y.append(list2[i][1])

    xline = xvals1
    yline = yvals1
    zline = xvals2

    ax.plot3D(xline, yline, zline, 'black')

    ax.scatter3D(listTab1X, listTab1Y, listTab2X, c=listTab2X)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show(fig)