from os import listdir
from os.path import isfile, join
import numpy as np
import cv2


def loadData() :
    mypath='/home/camelot/workspace/dicom-tracking-project/trainArtefacte/'
    onlyTrainfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    img = np.empty(len(onlyTrainfiles*2), dtype=object)
    data_Y = []
    data_X = []
    for n in range(0, 10):
        img[n] = cv2.imread( join(mypath,onlyTrainfiles[n]) )
        print(img[n][0])
        data_X = np.array([img])
        data_Y.append(1)

    # print(train_X.shape)


    mypath='/home/camelot/workspace/dicom-tracking-project/train/'
    onlyArtefactfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    for n in range(0, 10):
        img[len(onlyTrainfiles) + n] =  cv2.imread( join(mypath,onlyArtefactfiles[n]) )
        # ata_X = np.array(img[n])
        # data_X[len(onlyTrainfiles)+n] = cv2.imread( join(mypath,onlyArtefactfiles[n]) )
        data_Y.append(0)


    # 80% of the dataset for training
    split_percentage = 0.80

    split = int(split_percentage * len(data_X))

    # Train data set

    X_train = data_X[:split]

    y_train = data_Y[:split]

    # Test data set

    X_test = data_X[split:]

    y_test = data_Y[split:]

    print(data_X.shape)
    return X_train,y_train,X_test,y_test


loadData()
