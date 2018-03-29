# import tkinter.filedialog
from dicomConverter import dicom_converter
from draw_trajectory.tridTrajectory import draw3DTrajectorySmooth

file_name1 = '/home/julien/Images/DICOM/ARX1.rot.fa25.fr.rothschild.S.4674027.1_00000.DCM'
file_name2 = '/home/julien/Images/DICOM/ARX1.rot.7d6a.fr.rothschild.S.4674046.1_00000.DCM'


file_name3 = '/home/julien/Images/DICOM/ARX1.rot.1a4b.fr.rothschild.P.2104937.1_00000.DCM'

if __name__ == '__main__':
    # tab1 = dicom_converter(file_name3)


    tab1 = dicom_converter(file_name1)
    tab2 = dicom_converter(file_name2)

    # draw2DTrajectorySmooth(tab1)


    draw3DTrajectorySmooth(tab1, tab2)
    #
    # listTab1X = list()
    # listTab2X = list()
    # listTab1Y = list()
    # listTab2Y = list()
    #
    # for i in range(0, len(tab1)):
    #     listTab1X.append(tab1[i][0])
    #     listTab2X.append(tab2[i][0])
    #     listTab1Y.append(tab1[i][1])
    #     listTab2Y.append(tab2[i][1])
    #

    # draw3DTrajectory(listTab1X, listTab2X, listTab1Y, listTab2Y)
