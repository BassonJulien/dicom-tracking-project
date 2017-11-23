import PIL
import dicom
import os
import numpy
import Image
from matplotlib import pyplot, cm
import pylab



def save():
    os.system("ffmpeg -r 1 -i dicomImages/img%01d.png -vcodec mpeg4 -y movie.mp4")



PathDicom = "./dicom/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName, filename))


# Get ref file
RefDs = dicom.read_file(lstFilesDCM[0])

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.ImagerPixelSpacing[0]), float(RefDs.ImagerPixelSpacing[1]), float(1))


x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

# The array is sized based on 'ConstPixelDims'
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    # ArrayDicom[:, :, 1] = ds.pixel_array[0]

# pylab.imshow(ds.pixel_array[200], cmap=pylab.cm.bone)
ilo = 14

for j in range(0, ds.pixel_array.shape[0]):
    nameImg = '{0}{1}{2}'.format('dicomImages/img', j, '.png')
    img = Image.fromarray(ds.pixel_array[j])
    img = img.convert(mode='L')
    img.save(nameImg)

save()

# img.show()
# pylab.show()

# pyplot.figure(dpi=300)
# pyplot.axes().set_aspect('equal', 'datalim')
# pyplot.set_cmap(pyplot.gray())
# pyplot.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, 1]))