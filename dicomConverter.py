import dicom
import numpy as np
import skvideo.io
from orb_process import orbDetectionCrop


# Convert the dicom angio into a mp4 video*
# Take as input the path of the dicom to convert


def dicom_converter(file_name):
    ds = dicom.read_file(file_name)
    writer = skvideo.io.FFmpegWriter('./videos/uint8.avi')
    refPoint = [None,None]
    refPointTab = [None]
    print("taille tab",ds.pixel_array.shape[0])
    for j in range(0, ds.pixel_array.shape[0]):
        print("nouvellle iteration")
        frame = ds.pixel_array[j]
        outputdata = frame.astype(np.uint8)
        # outputdata,refPoint = orbDetection.main_orb_detection(outputdata,refPoint)
        if refPointTab[0] is None or len(refPointTab) < 2 :
            outputdata,refPoint = orbDetectionCrop.main_orb_detection(outputdata, refPoint, j)
            if refPointTab[0] is None :
                refPointTab = [refPoint]
            else :
                refPointTab.append(refPoint)

        if len(refPointTab) >= 2 :
            # Send the two last refPoints
            outputdata,refPoint = orbDetectionCrop.main_orb_detection(outputdata, refPointTab[j - 1:], j)
            refPointTab.append(refPoint)
        print("refpointab",refPointTab)
        writer.writeFrame(outputdata)
    writer.close()
    return refPointTab



