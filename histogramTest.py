import cv2
import numpy as np
from matplotlib import pyplot as plt
import dicom

file_name = '/home/camelot/Vidéos/angios/test1.DCM'
data_dicom = dicom.read_file(file_name)

img_dicom = np.array(data_dicom.pixel_array[0],np.uint8)

hist,bins = np.histogram(img_dicom.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img_dicom.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
