import dicom
import os
import numpy
import cv2
import sys
from matplotlib import pyplot, cm
import pylab



def save():
    os.system("ffmpeg -r 1 -i dicomImages/img%01d.png -vcodec mpeg4 -y movie.mp4")

#
#
# PathDicom = "./dicom/"
# lstFilesDCM = []  # create an empty list
# for dirName, subdirList, fileList in os.walk(PathDicom):
#     for filename in fileList:
#         if ".dcm" in filename.lower():  # check whether the file's DICOM
#             lstFilesDCM.append(os.path.join(dirName, filename))
#
#
# # Get ref file
# RefDs = dicom.read_file(lstFilesDCM[0])
#
# # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
# ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
# # Load spacing values (in mm)
# ConstPixelSpacing = (float(RefDs.ImagerPixelSpacing[0]), float(RefDs.ImagerPixelSpacing[1]), float(1))
#
#
# x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
# y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
# z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
#
# # The array is sized based on 'ConstPixelDims'
# ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
#
# # loop through all the DICOM files
# for filenameDCM in lstFilesDCM:
#     # read the file
#     ds = dicom.read_file(filenameDCM)
#     # store the raw image data
#     # ArrayDicom[:, :, 1] = ds.pixel_array[0]
#
# # pylab.imshow(ds.pixel_array[200], cmap=pylab.cm.bone)
# ilo = 14
#
# for j in range(0, ds.pixel_array.shape[0]):
#     nameImg = '{0}{1}{2}'.format('dicomImages/img', j, '.png')
#     img = Image.fromarray(ds.pixel_array[j])
#     img = img.convert(mode='L')
#     img.save(nameImg)
#
# save()

if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[1]

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()

# Read video
video = cv2.VideoCapture("dicom.mp4")

# Exit if video not opened.
if not video.isOpened():
    print ("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print ('Cannot read video file')
    sys.exit()

# Define an initial bounding box
bbox = (287, 23, 86, 320)

# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
