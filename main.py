import dicom
import numpy as np
import cv2
import orbDetection

file_name = '/home/camelot/Vidéos/angios/test1.DCM'
if __name__ == '__main__':
    # Data header of the dicom file
    data_dicom = dicom.read_file(file_name)
    img_dicom = np.array(data_dicom.pixel_array[0], np.uint8)
    orbDetection.main_orb_detection(img_dicom)
