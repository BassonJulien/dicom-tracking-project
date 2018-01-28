import tkinter.filedialog
from dicomConverter import dicom_converter
from plotPoints import draw_trajectory


pts = [(686, 324), (786, 374), (486, 474), (686, 343)]


from_filename = tkinter.filedialog.askopenfilename()
if from_filename != ():
    dicom_converter(from_filename)
draw_trajectory(pts)
