import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import model_from_json

from neuron_process import loadimages

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration var for neural network train
# More the batch size is higher more the algo will be performent
batch_size = 64
epochs = 100

num_classes = 2

class CatheterPredictor:
    def __init__(self, path_to_predictor_model='model.json', path_to_predictor_weights='model.h5'):
        self.model = None
        self._path_to_predictor_model = path_to_predictor_model
        self.path_to_predictor_weights = path_to_predictor_weights
        self.test_y = None
        self.test_x = None
        # create model
        self._create_model(self)
        try:
            #load model if already trained
            self._load_model(self)
        except IOError:
            # if not we need to train it
            self._train(self,self.model)

    @staticmethod
    def _prepare_train_dataset(self):
        train_X, train_Y, test_X, test_Y = loadimages.loading()
        print('Training data shape : ', train_X.shape, train_Y.shape)

        print('validation data shape : ', test_X.shape, test_Y.shape)
        classes = np.unique(train_Y)
        nClasses = len(classes)
        print('Total number of outputs : ', nClasses)
        print('Output classes : ', classes)

        train_X = train_X.reshape(-1, 150, 150, 1)
        test_X = test_X.reshape(-1, 150, 150, 1)
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

        train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2,
                                                                      random_state=13)

        self.test_y = test_Y
        self.test_x = test_X
        return   train_X, valid_X, train_label, valid_label, test_X, test_Y_one_hot

    # @staticmethod
    # def _prepare_test_dataset(self):

    @staticmethod
    def _create_model(self):

        # More the batch size is higher more the algo will be performent
        catheter_model = Sequential()
        catheter_model.add(
            Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(150, 150, 1), padding='same'))
        catheter_model.add(LeakyReLU(alpha=0.1))
        catheter_model.add(MaxPooling2D((2, 2), padding='same'))
        catheter_model.add(Dropout(0.12))

        catheter_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
        catheter_model.add(LeakyReLU(alpha=0.1))
        catheter_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        catheter_model.add(Dropout(0.12))

        catheter_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
        catheter_model.add(LeakyReLU(alpha=0.2))
        catheter_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        catheter_model.add(Dropout(0.18))

        catheter_model.add(Flatten())
        catheter_model.add(Dense(128, activation='linear'))
        catheter_model.add(LeakyReLU(alpha=0.1))
        catheter_model.add(Dropout(0.16))

        catheter_model.add(Dense(num_classes, activation='softmax'))
        catheter_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                               metrics=['accuracy'])
        catheter_model.summary()

        self.model = catheter_model

        return catheter_model

    @staticmethod
    def _load_model(self):

        # load json and create model
        json_file = open(self._path_to_predictor_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(self.path_to_predictor_weights)
        print("Loaded model from disk")
        self.model = loaded_model

    @staticmethod
    def _train(self,catheter_model):
        train_X, valid_X, train_label, valid_label, test_X, test_Y_one_hot = self._prepare_train_dataset(self)
        catheter_train = catheter_model.fit(train_X, train_label, batch_size=batch_size, epochs= epochs, verbose=1,
                                            validation_data=(valid_X, valid_label))
        # serialize model to JSON
        model_json = catheter_model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
            catheter_model.save_weights("model.h5")
        print("Saved model to disk")
        test_eval = catheter_model.evaluate(test_X, test_Y_one_hot, verbose=0)
        print('Test loss:', test_eval[0])
        accuracy = catheter_train.history['acc']
        val_accuracy = catheter_train.history['val_acc']
        loss = catheter_train.history['loss']
        val_loss = catheter_train.history['val_loss']
        epochs1 = range(len(accuracy))
        plt.plot(epochs1, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs1, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs1, loss, 'bo', label='Training loss')
        plt.plot(epochs1, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    @staticmethod
    def _prediction(self,test_x,test_y):
        predicted_classes = self.model.predict(test_x)
        predicted_classes = np.argmax(np.round(predicted_classes), axis=0)

        incorrect = np.where(predicted_classes != test_y)[0]
        correct = np.where(predicted_classes == test_y)[0]

        fig = plt.figure(figsize=(8, 8))
        columns = 5
        rows = 4

        print ("Found %d correct labels" % len(correct))
        for i, correct in enumerate(correct[:9]):
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(test_x[correct].reshape(150, 150))
            plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_y[correct]))
            # plt.tight_layout()

        print ("Found %d incorrect labels" % len(incorrect))
        for i, incorrect in enumerate(incorrect[:9]):
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(test_x[incorrect].reshape(150, 150))
            plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_y[incorrect]))
            # plt.tight_layout()

        plt.show()

    @staticmethod
    def _predict_byOneImage(self, test_x):
        test_x = test_x.reshape(-1, 150, 150, 1)
        test_x = test_x.astype('float32')
        test_x = test_x / 255.
        predicted_classe = self.model.predict(test_x)
        # print(predicted_classe)
        predicted_classe = np.argmax(np.round(predicted_classe), axis=1)

        print("predicted class",predicted_classe)
        return predicted_classe
        # incorrect = np.where(predicted_classes != test_y)[0]
        # correct = np.where(predicted_classes == test_y)[0]
        #
        # fig = plt.figure(figsize=(8, 8))
        # columns = 5
        # rows = 4
        #
        # print ("Found %d correct labels" % len(correct))
        # for i, correct in enumerate(correct[:9]):
        #     fig.add_subplot(rows, columns, i + 1)
        #     plt.imshow(test_x[correct].reshape(150, 150))
        #     plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_y[correct]))
        #     # plt.tight_layout()
        #
        # print ("Found %d incorrect labels" % len(incorrect))
        # for i, incorrect in enumerate(incorrect[:9]):
        #     fig.add_subplot(rows, columns, i + 1)
        #     plt.imshow(test_x[incorrect].reshape(150, 150))
        #     plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_y[incorrect]))
        #     # plt.tight_layout()
        #
        # plt.show()


