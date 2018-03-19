import numpy as np
import cv2
cap = cv2.VideoCapture('videos/dicom2.avi')


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()

r = cv2.selectROI(old_frame)
point = np.array((685, 340), dtype = np.int32)
point = np.array(point, dtype = np.float32)
p0 = np.float32(point).reshape(-1, 1, 2)

print(r)

# p0 = cv2.goodFeaturesToTrack(old_frame, mask = None, **feature_params)
# Create a mask image for drawing purposes

mask = np.zeros_like(old_frame)
# print(p0)

while 1:
    ret, frame = cap.read()
    frame_gray = cv2.medianBlur(frame, 3)

    frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None)
    # p1, st, err = cv2.calcOpticalFlowFarneback(frame, frame_gray, 0.5,1,3,15,3,5,1, 0)
    # print(p1)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    point = good_new.reshape(-1, 1, 2)
# cv2.destroyAllWindows()
cap.release()