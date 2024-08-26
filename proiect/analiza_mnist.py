import tensorflow as tf
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

from CNN import *


class Date_MNIST:
    import matplotlib.pyplot as plt
    def __init__(self):
        # Încarcă setul de date MNIST
        self._model = None
        self._mnist = tf.keras.datasets.mnist
        # X_train și y_train sunt datele de antrenament și etichetele corespunzătoare
        # X_test și y_test sunt datele de testare și etichetele corespunzătoare
        (self._X_train, self._y_train), (self._X_test, self._y_test) = self._mnist.load_data()

    def Hist_MNIST(self):
        # check if the data is uniform distributed over classes
        # Calculează frecvența fiecărei clase în setul de date
        classes, counts = np.unique(self._y_train, return_counts=True)
        # Calculează numărul de apariții pentru fiecare etichetă
        #label_counts = self._y_train.value_counts().sort_index()
        #plt.xticks(label_counts.index)
        # Afișează histograma
        plt.bar(classes, counts)
        plt.xlabel('Clasă (Cifra)')
        plt.ylabel('Numărul de Exemple')
        plt.title('Distribuția Claselor în Setul de Date MNIST')
        plt.xticks(classes)  # Etichetele de pe axa x sunt clasele (0-9)
        # Adaugă numărul de exemple în dreptul fiecărei bara
        for i, count in zip(classes, counts):
            plt.text(i, count, str(count), ha='center', va='bottom')

        plt.show()

    def Img_MNIST(self):
        # Afișează primele câteva imagini și etichetele lor corespunzătoare
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(self._X_train[i], cmap='gray')
            plt.title(f'Label: {self._y_train[i]}')
            plt.axis('off')
        plt.show()

    def Normalize_MNIST(self):
        # normalise
        self._X_train, self._X_test = self._X_train / 255, self._X_test / 255

        # transform labels into one-hot encoding
        self._y_trainhot = tf.keras.utils.to_categorical(self._y_train, 10)
        self._y_testhot = tf.keras.utils.to_categorical(self._y_test, 10)

        # input image dimensions
        # loaded mnist data are 28x28 images, but CNN requires new reshapes (28x28x1)
        img_rows, img_cols = 28, 28
        self._X_train = self._X_train.reshape(self._X_train.shape[0], img_rows, img_cols, 1)
        self._X_test = self._X_test.reshape(self._X_test.shape[0], img_rows, img_cols, 1)
        self._input_shape = (img_rows, img_cols, 1)

    def Train_MNIST_Adam(self, _batch_size=128, _epochs=15):
        self._model = ModelCNN(self._X_train, self._y_train, self._X_test, self._y_test, self._input_shape,
                               batch_size=_batch_size, epochs=_epochs, optimizer=Adam(),
                               loss=sparse_categorical_crossentropy, metrics=['accuracy'])
        self._model.CreateCNN()
        return self._model

    def Train_MNIST_SGD(self, _batch_size=128, _epochs=15):
        self._model = ModelCNN(self._X_train, self._y_trainhot, self._X_test, self._y_testhot, self._input_shape,
                               batch_size=_batch_size, epochs=_epochs, optimizer=tf.keras.optimizers.SGD(),
                               loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
        self._model.CreateCNN()
        return self._model

    def Test_MNIST(self, x_test_normalized):
        predictions_one_hot = self._model.prediction(x_test_normalized)
        predictions_one_show = predictions_one_hot
        print('predictions_one_hot:', predictions_one_hot.shape)
        #print(predictions_one_hot)
        predictions = np.argmax(predictions_one_hot)
        #print(predictions )
        predictions = np.argmax(predictions_one_hot, axis=1)
        #print(predictions )
        frame = pd.DataFrame(predictions, columns=['digit'])
        #tiparesc primele 5 rezult
        print(frame.head())
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        for i in range(16):
            img = x_test_normalized[i].reshape(28, 28)  # Accesează direct imaginea din tensorul normalizat
            #img = x_test_normalized[i].numpy().reshape(28, 28)
            axes[i].imshow(img, cmap='gray')

            axes[i].set_title(f'Predicted: {predictions[i]}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
