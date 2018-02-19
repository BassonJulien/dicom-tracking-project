import dicom
import numpy as np
import skvideo.io
import PIL
import cv2
from matplotlib import pyplot as plt




# Convert the dicom angio into a mp4 video*
# Take as input the path of the dicom to convert


def dicom_converter(path):
    ds = dicom.read_file("/home/julien/Images/DICOM/ARX1.rot.fa25.fr.rothschild.S.4674027.1_00000.DCM")
    writer = skvideo.io.FFmpegWriter('videos/dicom7.avi')
    print len(ds.pixel_array)
    plt.imshow(ds.pixel_array[200])
    plt.show()
    cv2.waitKey(0)
    # for j in range(0, ds.pixel_array.shape[0]):
    #     frame = ds.pixel_array[j]
    #     outputdata = frame.astype(np.uint8)
    #     writer.writeFrame(outputdata)
    # writer.close()


