from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import cv2

import loadimages

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_X,train_Y,test_X,test_Y = loadimages.loading()
print('Training data shape : ', train_X.shape, train_Y.shape)

print('Validation data shape : ', test_X.shape, test_Y.shape)
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


train_X = train_X.reshape(-1, 150,150, 1)
test_X = test_X.reshape(-1, 150,150, 1)
train_X.shape, test_X.shape
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
# print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

# More the batch size is higher more the algo will be performent
batch_size = 64
epochs = 20
num_classes = 2

catheter_model = Sequential()
catheter_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(150,150,1),padding='same'))
catheter_model.add(LeakyReLU(alpha=0.1))
catheter_model.add(MaxPooling2D((2, 2),padding='same'))
catheter_model.add(Dropout(0.1))

catheter_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
catheter_model.add(LeakyReLU(alpha=0.1))
catheter_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
catheter_model.add(Dropout(0.1))

catheter_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
catheter_model.add(LeakyReLU(alpha=0.2))
catheter_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
catheter_model.add(Dropout(0.1))

catheter_model.add(Flatten())
catheter_model.add(Dense(128, activation='linear'))
catheter_model.add(LeakyReLU(alpha=0.1))
catheter_model.add(Dropout(0.15))

catheter_model.add(Dense(num_classes, activation='softmax'))
catheter_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
catheter_model.summary()

catheter_train = catheter_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
test_eval = catheter_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])



predicted_classes = catheter_model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

incorrect = np.where(predicted_classes!=test_Y)[0]
correct = np.where(predicted_classes == test_Y)[0]

fig=plt.figure(figsize=(8, 8))
columns = 5
rows = 4

print ("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[correct].reshape(150,150), cmap='gray')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()
plt.show()

print ("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_X[incorrect].reshape(150, 150), cmap='gray')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()
plt.show()

accuracy = catheter_train.history['acc']
val_accuracy = catheter_train.history['val_acc']
loss = catheter_train.history['loss']
val_loss = catheter_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# print ("Found %d correct labels" % len(correct))
# for i, correct in enumerate(correct):
#     cv2.imwrite('/home/camelot/workspace/dicom-tracking-project/goodResult/goodResult(%d).png' % i, test_X[correct].reshape(150,150)*250)
#
# print ("Found %d incorrect labels" % len(incorrect))
# for i, incorrect in enumerate(incorrect):
#     cv2.imwrite('/home/camelot/workspace/dicom-tracking-project/badResult/badResult(%d).png' % i, test_X[incorrect].reshape(150, 150)*250)
#
