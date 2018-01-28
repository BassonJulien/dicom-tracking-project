import tkinter.filedialog
from dicomConverter import dicom_converter

from_filename = tkinter.filedialog.askopenfilename()
print (from_filename)
dicom_converter(from_filename)
