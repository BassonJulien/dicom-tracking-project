# import tkinter.filedialog
from dicomConverter import dicom_converter
from plotPoints import draw_trajectory, lissage


pts = [(15, 15), (300, 50), (100, 500), (686, 324), (786, 374), (886, 474), (986, 543)]


# from_filename = tkinter.filedialog.askopenfilename()
# if from_filename != ():

# dicom_converter(from_filename)

# dicom_converter("")




# draw_trajectory(pts)
# ptx = [12, 100, 686, 786, 886, 986]
# pty = [15, 500, 324, 374, 474, 543]
# pts2 = [(100, 257), (686, 412), (786, 349), (886, 424)]

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
import dicom
import numpy as np
import cv2
import orbDetection

# file_name = '/home/camelot/Vidéos/angios/test1.DCM' #marche
# file_name = '/home/camelot/Vidéos/angios/test2.DCM' #marche
file_name = '/home/camelot/Vidéos/angios/ARX1.rot.1a49.fr.rothschild.S.4818696.1_00000.DCM' #nope
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.8e1c.fr.rothschild.S.4811827.1_00000.DCM'
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.fa25.fr.rothschild.S.4674027.1_00000.DCM' #normal photo
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.fc74.fr.rothschild.S.4925457.1_00000.DCM'
# New dicom
# file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.96ad.fr.rothschild.S.4811821.1_00000.DCM'
# file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.826d.fr.rothschild.S.4818854.1_00000.DCM' #marche
if __name__ == '__main__':
    dicom_converter(file_name)

    # data_dicom = dicom.read_file(file_name)
    # img_dicom = np.array(data_dicom.pixel_array[0], np.uint8)
    # orbDetection.main_orb_detection(img_dicom)
    # dicom_converter("cc")

