import dicom
import numpy as np
import skvideo.io


def dicom_converter(path):
    ds = dicom.read_file(path)
    writer = skvideo.io.FFmpegWriter('output.mp4')

    for j in range(0, ds.pixel_array.shape[0]):
        frame = ds.pixel_array[j]
        outputdata = frame.astype(np.uint8)
        writer.writeFrame(outputdata)
    writer.close()


