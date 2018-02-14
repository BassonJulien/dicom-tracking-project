import dicom
import numpy as np
import skvideo.io

# Convert the dicom angio into a mp4 video*
# Take as input the path of the dicom to convert


def dicom_converter(path):
    ds = dicom.read_file("/home/julien/Images/DICOM/ARX1.rot.1a4b.fr.rothschild.P.2104937.1_00000.DCM")
    writer = skvideo.io.FFmpegWriter('uint8.avi')

    for j in range(0, ds.pixel_array.shape[0]):
        frame = ds.pixel_array[j]
        outputdata = frame.astype(np.uint8)
        writer.writeFrame(outputdata)
    writer.close()


