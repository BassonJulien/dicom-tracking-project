from PIL import Image
import numpy as np
import glob

# Artefact and catheter folder
imageFolderPathCatheter = '/home/camelot/workspace/dicom-tracking-project/train/'
imageFolderPathArtefact = '/home/camelot/workspace/dicom-tracking-project/trainArtefacte/'
imagePathArtefact = glob.glob(imageFolderPathArtefact + '/*.png')
imagePathCatheter = glob.glob(imageFolderPathCatheter + '/*.png')

def loading () :
    data_x = []
    data_y = []

    # Insert train artefact set images
    for path in imagePathArtefact :
        img = np.array(Image.open(path))
        # print(img)
        if img.shape ==(150,150):
            data_x.append(img)
            data_y.append(0)


    # Insert train catheter set images
    for path in imagePathCatheter :
        img = np.array(Image.open(path))
        # print(img)
        if img.shape ==(150,150):
            data_x.append(img)
            data_y.append(1)



    # 80% of the dataset for training
    split_percentage = 0.80

    split = int(split_percentage * len(data_x))

    # Train data set

    X_train = np.array(data_x[:split])

    y_train = np.array(data_y[:split])

    # Test data set

    X_test = np.array(data_x[split:])

    y_test = np.array(data_y[split:])

    return X_train, y_train, X_test, y_test

imageFolderPathCatheter = '/home/camelot/workspace/dicom-tracking-project/testData/'
imageFolderPathArtefact = '/home/camelot/workspace/dicom-tracking-project/testData/'
imagePathArtefact = glob.glob(imageFolderPathArtefact + '/*.png')
imagePathCatheter = glob.glob(imageFolderPathCatheter + '/*.png')

def loadingTest () :
    data_x = []
    data_y = []

    # Insert train artefact set images
    for path in imagePathArtefact :
        img = np.array(Image.open(path))
        # print(img)
        if img.shape ==(150,150):
            data_x.append(img)
            data_y.append(0)


    # Insert train catheter set images
    for path in imagePathCatheter :
        img = np.array(Image.open(path))
        # print(img)
        if img.shape ==(150,150):
            data_x.append(img)
            data_y.append(1)



    # 80% of the dataset for training
    split_percentage = 0.80

    split = int(split_percentage * len(data_x))

    # Train data set

    X_train = np.array(data_x[:split])

    y_train = np.array(data_y[:split])

    # Test data set

    X_test = np.array(data_x[split:])

    y_test = np.array(data_y[split:])

    return X_train, y_train, X_test, y_test




# temp = np.array([[[np.array(lignes)] for lignes in img]for img in im_array])
