import cv2
#from tqdm import tqdm
from tensorflow.keras.utils import *
import keras
from keras import layers
from CNN import *
#from sklearn.model_selection import train_test_split
import math
import datetime
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore") # to clean up output cells

class Hand_CSV:

    def __init__(self, path_dir_train, path_dir_test, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS):
        # importing data

        self._path_dir_train = path_dir_train
        self._path_dir_test = path_dir_test
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self._test = None
        if not os.path.exists(self._path_dir_test):
            os.makedirs(self._path_dir_test)
        current_directory = os.getcwd()  # Obține directorul curent de lucru
        self._file_test_path_csv = os.path.join(current_directory,path_dir_test,"hand.csv")  # Creează calea completă către fișierul de bază de date
        # Verifică dacă fișierul CSV există deja
        exista_fisier = os.path.exists(self._file_test_path_csv)
        # Dacă fișierul există, șterge-l
        if exista_fisier:
            os.remove(self._file_test_path_csv)
            # Creează un fișier CSV gol
        #open(file_path_csv, 'a').close()

        # Deschide sau creează fișierul CSV în modul de adăugare ('a')
        with open(self._file_test_path_csv, 'a') as file:

         # Iterare prin fiecare imagine din director
            for filename in os.listdir(self._path_dir_train):
                if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):
                    image_path = os.path.join(self._path_dir_train, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image_uint8 = (image * 255).astype(np.uint8)

                    # Aplică denoise
                    image = cv2.fastNlMeansDenoising(image_uint8, None, h=10, templateWindowSize=7, searchWindowSize=21)

                    # Binarizează imaginea
                    _, image_binarizată = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

                    # Redimensionează imaginea
                    image_redimensionată = cv2.resize(image_binarizată, (self.IMAGE_WIDTH , self.IMAGE_HEIGHT))

                    #plt.subplot(1, 2, 1)
                    #plt.imshow(image_redimensionată, cmap='gray')
                    #plt.title('Imagine originală')
                    #plt.show()
                    # Transformă imaginea într-un vector
                    vector_pixeli = image_redimensionată.flatten()
                    # Elimină primul pixel din fiecare imagine (dacă este necesar)
                    #vector_pixeli = vector_pixeli[1:]
                    # Adaugă eticheta (opțional)
                    #etichetă = 0  # Exemplu de etichetă
                    #vector_pixeli = np.insert(vector_pixeli, 0, etichetă)

                    # Creează un DataFrame Pandas
                    df = pd.DataFrame(vector_pixeli.reshape(1, -1))
                    # Salvează în fișierul CSV fără antet și index

                    #df.to_csv(file, header=False, index=False)
                    text = df.to_csv(header=False, index=False)
                    text = text.replace('\n', '')

                    # Scriem reprezentarea text în fișierul CSV
                    file.write(text)
            file.close()

            self._test = pd.read_csv(self._file_test_path_csv)
            self._X_test = self._test.iloc[:, 0:784]
            num_images = len(self._X_test)
            print(num_images)
            num_rows = 5  # Ajustăm numărul de rânduri în funcție de numărul de imagini
            num_cols = 4  # Ajustăm numărul de coloane în funcție de numărul de imagini

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
            axes = axes.flatten()
            for i in range(min(20, num_images)):
                img = self._X_test.iloc[i, :].values.reshape(self.IMAGE_HEIGHT,
                                                             self.IMAGE_WIDTH)  # Extragem imaginea din datele de testare
                axes[i].imshow(img, cmap='gray')
                axes[i].axis('off')
            for i in range(num_images, num_rows * num_cols):
                axes[i].axis('off')  # ascunde celelalte subploturi
            plt.tight_layout()
            # Setează titlul pentru figura
            # Setează numele pentru figura
            fig.set_label('Imagini de test scrise de mana')

            fig.suptitle('Imagini de test scrise de mana', fontsize=12)
            plt.show()


    def Normalize_CSV(self):

        #self._test = pd.read_csv(self._file_test_path_csv)
        #self._X_test = self._test.iloc[:, 0:784]
        x_test_re = self._test.to_numpy().reshape(self._X_test.shape[0], 28, 28)
        x_test_with_chanels = x_test_re.reshape(
            x_test_re.shape[0],
            self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS
        )
        x_test_normalized = x_test_with_chanels / 255
        return x_test_normalized




