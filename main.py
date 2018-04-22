# import tkinter.filedialog
from dicomConverter import dicom_converter

file_name1 = '/home/julien/Images/DICOM/ARX1.rot.fa25.fr.rothschild.S.4674027.1_00000.DCM'
file_name2 = '/home/julien/Images/DICOM/ARX1.rot.7d6a.fr.rothschild.S.4674046.1_00000.DCM'


file_name3 = '/home/julien/Images/DICOM/ARX1.rot.1a4b.fr.rothschild.P.2104937.1_00000.DCM'

# from plotPoints import draw_trajectory, lissage
#
#
# pts = [(15, 15), (300, 50), (100, 500), (686, 324), (786, 374), (886, 474), (986, 543)]
#
#
# # from_filename = tkinter.filedialog.askopenfilename()
# # if from_filename != ():
#
# # dicom_converter(from_filename)
#
# dicom_converter("")
#
#
#
#
# draw_trajectory(pts)
# ptx = [12, 100, 686, 786, 886, 986]
# pty = [15, 500, 324, 374, 474, 543]
# pts2 = [(100, 257), (686, 412), (786, 349), (886, 424)]
#
# # draw_trajectory(pts)
# # liste = lissage(pts, 1)
# #
# # draw_trajectory(liste)

# file_name = '/home/camelot/Vidéos/angios/test1.DCM' #marche
# file_name = '/home/camelot/Vidéos/angios/test2.DCM' #marche
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.1a49.fr.rothschild.S.4818696.1_00000.DCM' #marche
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.8e1c.fr.rothschild.S.4811827.1_00000.DCM' #marche
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.fa25.fr.rothschild.S.4674027.1_00000.DCM' #normal photo marche
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.fc74.fr.rothschild.S.4925457.1_00000.DCM'
# New dicom
# file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.96ad.fr.rothschild.S.4811821.1_00000.DCM'
# file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.826d.fr.rothschild.S.4818854.1_00000.DCM' #marche
# file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.1211.fr.rothschild.S.4676231.1_00000.DCM' #nope
file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.1a4b.fr.rothschild.P.2104937.1_00000.DCM' #marche
# file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.cdc1.fr.rothschild.P.2104952.1_00000.DCM'
if __name__ == '__main__':
    # tab1 = dicom_converter(file_name3)
    dicom_converter(file_name)
    #
    # tab1 = dicom_converter(file_name1)
    # tab2 = dicom_converter(file_name2)
    #
    # # draw2DTrajectorySmooth(tab1)
    #
    #
    # draw3DTrajectorySmooth(tab1, tab2)
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
