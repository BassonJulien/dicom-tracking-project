import dicom
import numpy as np
import cv2

# Convert the dicom angio into a mp4 video*
# Take as input the path of the dicom to convert
crop_constante = 75
x_origin = 200
y_origin = 200


def create_artefacte(file_name):
    global x_origin,y_origin

    ds = dicom.read_file(file_name)
    for j in range(0, ds.pixel_array.shape[0]):
        frame = ds.pixel_array[j]
        img_dicom = frame.astype(np.uint8)
        pointY1_crop = -crop_constante+y_origin
        pointY2_crop = crop_constante+y_origin
        pointX1_crop = -crop_constante+x_origin
        pointX2_crop = crop_constante+x_origin
        print(pointY1_crop,pointY2_crop,pointX2_crop,pointX1_crop)
        img_dicom = img_dicom[pointY1_crop:pointY2_crop, pointX1_crop:pointX2_crop]
        x_origin += 50
        if x_origin > 900:
            x_origin = 75
            y_origin += 50
        cv2.imwrite(
            '/home/camelot/workspace/dicom-tracking-project/trainArtefacte/artefacte4(%d).png' % j,
            img_dicom)


# data_dicom = dicom.read_file(file_name)
#     img_dicom = np.array(data_dicom.pixel_array[0], np.uint8)
#     orbDetection.main_orb_detection(img_dicom)
#     dicom_converter("cc")
# file_name = '/home/camelot/Vidéos/angios/test1.DCM'
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.1a49.fr.rothschild.S.4818696.1_00000.DCM'
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.8e1c.fr.rothschild.S.4811827.1_00000.DCM'
# file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.826d.fr.rothschild.S.4818854.1_00000.DCM' #marche

file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.1a4b.fr.rothschild.P.2104937.1_00000.DCM'
create_artefacte(file_name)
