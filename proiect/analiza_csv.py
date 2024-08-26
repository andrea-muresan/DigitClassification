import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
#from skimage.transform import resize
import cv2
#from tqdm import tqdm
from tensorflow.keras.utils import *
import keras
from keras import layers
from CNN import *
from sklearn.model_selection import train_test_split
import math
import datetime

warnings.filterwarnings("ignore") # to clean up output cells

class Date_CSV:

    def __init__(self, path_file_train, path_file_test, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS):
        # importing data
        self._path_file_train = path_file_train
        self._path_file_test = path_file_test
        self._train = pd.read_csv(self._path_file_train)
        self._test = pd.read_csv(self._path_file_test)
        self._model = None
        print(f'Train Data:\n {self._train.head()}')
        print(f'Test Data: \n {self._test.head()}')
        print(f'\n Informatii Date de Test:\n {self._train.info()}, Date de Test(Numar linii si Coloane):\n {self._train.shape}')

        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.IMAGE_CHANNELS = IMAGE_CHANNELS

        #selectez set date
        self._X = self._train.iloc[:, 1:785]
        self._y = self._train.iloc[:, 0]

        self._X_test = self._test.iloc[:, 0:784]

        #impart datele pe test 20% train 80%
        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(self._X, self._y, test_size=0.2,  random_state=42) #random_state=1212)
        # afisez histograma
        self.plot_train_histogram()

        # afisez esantion de 16
        self.plot_images_with_labels()

        self._image_data = []

    def Normalize_CSV_Test(self):
        x_test_re = self._test.to_numpy().reshape(self._X_test.shape[0], 28, 28)
        x_test_with_chanels = x_test_re.reshape(
            x_test_re.shape[0],
            self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS
        )
        x_test_normalized = x_test_with_chanels / 255
        return x_test_normalized
    def Train_CSV(self,_batch_size=64, _epochs=15):
        x_train_normalized, y_train_labels, x_validation_normalized, y_validation_labels, x_test_normalized = self.shape_normalization()
        img_rows, img_cols = 28, 28

        input_shape = (img_rows, img_cols, 1)

        self._model = ModelCNN(x_train_normalized, y_train_labels, x_validation_normalized, y_validation_labels, input_shape,batch_size=_batch_size, epochs=_epochs, optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['accuracy'])
        self._model.CreateCNN()
        return self._model, x_test_normalized

    def Test_CSV(self,x_test_normalized):
        predictions_one_hot = self._model.prediction(x_test_normalized)
        predictions_one_show = predictions_one_hot
        print('predictions_one_hot:', predictions_one_hot.shape)
        predictions = np.argmax(predictions_one_hot, axis=1)
        frame = pd.DataFrame(predictions, columns=['digit'])
        print(frame.head())
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        for i in range(16):
            img = self._X_test.iloc[i, :].values.reshape(self.IMAGE_HEIGHT,
                                                         self.IMAGE_WIDTH)  # Extragem imaginea din datele de testare
            axes[i].imshow(img, cmap='gray')

            axes[i].set_title(f'Predicted: {predictions[i]}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
        return

    def Test_CSV_Hand(self,x_test_normalized):
        predictions_one_hot = self._model.prediction(x_test_normalized)
        predictions_one_show = predictions_one_hot
        print('predictions_one_hot:', predictions_one_hot.shape)
        predictions = np.argmax(predictions_one_hot, axis=1)
        frame = pd.DataFrame(predictions, columns=['digit'])
        # tiparesc primele 5 rezult
        print(frame.head())
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        for i in range(16):
            img = x_test_normalized[i].reshape(28, 28)  # Accesează direct imaginea din tensorul normalizat
            axes[i].imshow(img, cmap='gray')

            axes[i].set_title(f'Predicted: {predictions[i]}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
        return


    def plot_images_with_labels(self, num_images=16):
        # Afișează imagini cu etichetele lor asociate

        x_train_re = self._X_train.to_numpy().reshape(-1, 28,28)  # -1 specifică că numpy trebuie să calculeze dimensiunea acestei axe
        y_train_re = self._y_train.values

        numbers_to_display = min(num_images,len(x_train_re))  # Asigură-te că nu încerci să afișezi mai multe imagini decât ai disponibile
        num_cells = math.ceil(math.sqrt(numbers_to_display))
        plt.figure(figsize=(10, 10))
        for i in range(numbers_to_display):
            plt.subplot(num_cells, num_cells, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(x_train_re[i], cmap=plt.cm.binary)
            plt.xlabel(f'Label:{y_train_re[i]}')
        plt.show()

    def plot_train_histogram(self):
        # Calculează numărul de apariții pentru fiecare etichetă
        label_counts = self._y_train.value_counts().sort_index()

        # Afișează numărul de apariții pentru fiecare etichetă sub formă de bare
        bars = plt.bar(label_counts.index, label_counts.values, color='skyblue')
        plt.title('Numărul de date aferente fiecărei cifre')
        plt.xlabel('Cifra')
        plt.ylabel('Număr de apariții')
        plt.xticks(label_counts.index)
        #plt.yticks(label_counts.values)
        plt.grid(axis='y')
        plt.bar_label(bars)
        plt.show()

    def shape_normalization(self):

        x_train_re = self._X_train.to_numpy().reshape(self._X_train.shape[0], 28, 28)
        y_train_re = self._y_train.values
        x_validation_re = self._X_val.to_numpy().reshape(self._X_val.shape[0], 28, 28)
        y_validation_re = self._y_val.values
        x_test_re = self._test.to_numpy().reshape(self._X_test.shape[0], 28, 28)

        x_train_with_chanels = x_train_re.reshape(
            x_train_re.shape[0],
            self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS
        )

        x_validation_with_chanels = x_validation_re.reshape(
            x_validation_re.shape[0],
            self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS
        )

        x_test_with_chanels = x_test_re.reshape(
            x_test_re.shape[0],
            self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS
        )

        # normalizez imaginea
        x_train_normalized = x_train_with_chanels / 255
        x_validation_normalized = x_validation_with_chanels / 255
        x_test_normalized = x_test_with_chanels / 255
        #y_train_labels = tf.keras.utils.to_categorical(y_train_re, 10)
        #y_validation_labels = tf.keras.utils.to_categorical(y_validation_re, 10)
        y_train_labels = y_train_re
        y_validation_labels = y_validation_re

        return x_train_normalized, y_train_labels, x_validation_normalized, y_validation_labels, x_test_normalized


    '''
    def plot_images_with_labels(self, num_images=16):
        # Afișează imagini cu etichetele lor asociate
        x_train_re = self._X_train.to_numpy().reshape(self._X_train.shape, 28, 28)
        y_train_re = self._y_train.values
        x_validation_re = self._X_val.to_numpy().reshape(self._X_val.shape, 28, 28)
        y_validation_re = self._y_val.values
        x_test_re = self._test.to_numpy().reshape(self._X_test.shape, 28, 28)

        numbers_to_display = num_images
        num_cells = math.ceil(math.sqrt(numbers_to_display))
        plt.figure(figsize=(20, 20))
        for i in range(numbers_to_display):
            plt.subplot(num_cells, num_cells, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(x_train_re[i], cmap=plt.cm.binary)
            plt.xlabel(y_train_re[i])
        plt.show()

    def Get_Train_Data(self, datafile):
        feature, label = datafile.drop("label", axis=1), datafile[["label"]]
        return feature, label

    def Normalize_Image(self):
        for image_row in tqdm(self._train.loc[:5000].index):
            tmp = np.array(list(self._X_train.loc[image_row]))
            #print(tmp)
            tmp = np.resize(tmp, (28, 28))
            img_r = resize(tmp, (28, 28, 1))
            #image = tf.image.convert_image_dtype(img_r, dtype=tf.float32)
            self._image_data.append(img_r)

        plt.figure(figsize=(10, 10))

        for i in range(16):
            image = self._image_data[i]
            plt.subplot(4, 4, i + 1)
            plt.imshow(image)
            plt.axis('off')

        plt.show()
        train_labels = tf.keras.utils.to_categorical(self._Y_train, 10)
    '''




    def test(self):
        input_shape = (28, 28, 1)
        #cnn = ModelCNN(self._X_train_norm,self._y_train,self._X_test_norm,y_testhot,input_shape)
        #cnn.CreateCNN()

