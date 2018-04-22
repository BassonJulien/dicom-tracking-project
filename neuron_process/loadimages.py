from PIL import Image
import numpy as np
import glob

# Artefact and catheter folder
imageFolderPathCatheter = './train/'
imageFolderPathArtefact = './trainArtefacte/'
imagePathArtefact = glob.glob(imageFolderPathArtefact + '/*.png')
imagePathCatheter = glob.glob(imageFolderPathCatheter + '/*.png')

def loading () :
    data_x = []
    data_y = []
    dataArte_x = []
    dataArte_y = []

    dataTrain_x = []
    dataTrain_y = []

    # Insert train artefact set images
    for path in imagePathArtefact :
        img = np.array(Image.open(path))
        # print(img)
        if img.shape ==(150,150):
            dataArte_x.append(img)
            dataArte_y.append(0)


    # Insert train catheter set images
    for path in imagePathCatheter :
        img = np.array(Image.open(path))
        # print(img)
        if img.shape ==(150,150):
            dataTrain_x.append(img)
            dataTrain_y.append(1)

    for i in range(0, len(dataTrain_x)):
        if i % 2 == 0:
            if img.shape == (150, 150):
                data_x.append(dataTrain_x[i])
                data_y.append(dataTrain_y[i])
        else:
            if img.shape == (150, 150):
                data_x.append(dataArte_x[i])
                data_y.append(dataArte_y[i])

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

imageFolderPathCatheterTest = './testData/'
imageFolderPathArtefactTest = './testDataArte/'
imagePathArtefactTest = glob.glob(imageFolderPathArtefactTest + '/*.png')
imagePathCatheterTest = glob.glob(imageFolderPathCatheterTest + '/*.png')

def loadingTest (img) :
    data_x = []
    data_y = []
    dataArte_x = []
    dataArte_y = []

    dataTrain_x = []
    dataTrain_y = []

    # Insert train artefact set images
    for path in imagePathArtefact:
        img = np.array(Image.open(path))
        # print(img)
        if img.shape == (150, 150):
            dataArte_x.append(img)
            dataArte_y.append(0)

    # Insert train catheter set images
    for path in imagePathCatheter:
        img = np.array(Image.open(path))
        # print(img)
        if img.shape == (150, 150):
            dataTrain_x.append(img)
            dataTrain_y.append(1)

    for i in range(0, len(dataTrain_x)):
        if i % 2 == 0:
            if img.shape == (150, 150):
                data_x.append(dataTrain_x[i])
                data_y.append(dataTrain_y[i])
        else:
            if img.shape == (150, 150):
                data_x.append(dataArte_x[i])
                data_y.append(dataArte_y[i])

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
