import numpy as np
import neuron_preprocess
import dicom
import skvideo.io
from catheter_predictor import CatheterPredictor
# file_name = '/home/camelot/Vidéos/angios/test1.DCM' #marche
# file_name = '/home/camelot/Vidéos/angios/test2.DCM' #marche
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.1a49.fr.rothschild.S.4818696.1_00000.DCM' #marche
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.8e1c.fr.rothschild.S.4811827.1_00000.DCM' #marche
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.fa25.fr.rothschild.S.4674027.1_00000.DCM' #normal photo marche
# file_name = '/home/camelot/Vidéos/angios/ARX1.rot.fc74.fr.rothschild.S.4925457.1_00000.DCM'
# New dicom
# file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.96ad.fr.rothschild.S.4811821.1_00000.DCM'
# file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.826d.fr.rothschild.S.4818854.1_00000.DCM' #marche
# file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.1211.fr.rothschild.S.4676231.1_00000.DCM' #nope
# file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.1a4b.fr.rothschild.P.2104937.1_00000.DCM' #marche
# file_name = '/home/camelot/Vidéos/angios2/ARX1.rot.cdc1.fr.rothschild.P.2104952.1_00000.DCM'

def dicom_converter(file_name):
    print("-----------------------------------------------------------------------------",file_name)
    model = CatheterPredictor()
    # model = None
    ds = dicom.read_file(file_name)
    writer = skvideo.io.FFmpegWriter('../videos/uint8.avi')
    refPoint = [None,None]
    refPointTab = [None]
    print("taille tab",ds.pixel_array.shape[0])
    for j in range(0, ds.pixel_array.shape[0]):
        print("nouvellle iteration")
        frame = ds.pixel_array[j]
        outputdata = frame.astype(np.uint8)
        # outputdata,refPoint = orbDetection.main_orb_detection(outputdata,refPoint)
        if refPointTab[0] is None or len(refPointTab) < 2 :
            outputdata,refPoint = neuron_preprocess.detection_process(outputdata, refPoint, j, model)
            if refPointTab[0] is None :
                refPointTab = [refPoint]
            else :
                refPointTab.append(refPoint)

        if len(refPointTab) >= 2 :
            # Send the two last refPoints
            outputdata,refPoint = neuron_preprocess.detection_process(outputdata, refPointTab[j - 1:], j, model)
            refPointTab.append(refPoint)
        print("refpointab",refPointTab)
        writer.writeFrame(outputdata)
    writer.close()


# catheter_predictor
# dicom_converter(file_name)

