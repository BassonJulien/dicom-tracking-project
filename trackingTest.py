import cv2
import sys
import numpy as np
from plotPoints import draw_trajectory, lissage


def preprocess(frame):

    frame = cv2.medianBlur(frame, 5)
    # ret, frame = cv2.threshold(frame, 114, 180, cv2.THRESH_BINARY_INV)

    # Dilatation
    # kernel_dilate = np.ones((6, 5), np.uint8)
    # frame = cv2.dilate(frame, kernel_dilate, iterations=1)
    #
    # # Erosion
    # kernel_erode = np.ones((3, 1), np.uint8)
    # frame = cv2.erode(frame, kernel_erode, iterations=1)

    # frame = cv2.Canny(frame, 100, 200)

    return frame


def main():

    listX = list()
    listY = list()

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[1]

    if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    # Read video
    video = cv2.VideoCapture("videos/dicom1.avi")

    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    frame = preprocess(frame)



    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Pre proccessing to reduce noise on the images
        frame = preprocess(frame)

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

            x = (bbox[0] + bbox[2]/2)
            y = bbox[1] + bbox[3]/2

            listX.append(x)
            listY.append(y)

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

    return listX, listY


x, y = main()

listeXY = list()


for i in range(0, len(x)):
    listTransition = []
    listTransition.append(x[i])
    listTransition.append(y[i])
    listeXY.append(listTransition)


draw_trajectory(listeXY)
lissage(listeXY, 5)
lissage(listeXY, 15)
lissage(listeXY, 35)
lissage(listeXY, 55)
lissage(listeXY, 75)
lissage(listeXY, 750)
lissage(listeXY, 7500)

