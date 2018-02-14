import tkinter.filedialog
from dicomConverter import dicom_converter
from plotPoints import draw_trajectory, lissage


pts = [(15, 15), (300, 50), (100, 500), (686, 324), (786, 374), (886, 474), (986, 543)]


# from_filename = tkinter.filedialog.askopenfilename()
# if from_filename != ():

# dicom_converter(from_filename)

dicom_converter("")




draw_trajectory(pts)
ptx = [12, 100, 686, 786, 886, 986]
pty = [15, 500, 324, 374, 474, 543]
pts2 = [(100, 257), (686, 412), (786, 349), (886, 424)]

# draw_trajectory(pts)
# liste = lissage(pts, 1)
#
# draw_trajectory(liste)
