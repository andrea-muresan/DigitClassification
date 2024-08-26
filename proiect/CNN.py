import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.initializers import VarianceScaling  # Layer weight initializers
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sn


class ModelCNN:

    def __init__(self, Xtrain_data, Ytrain_data,
                 Xtest_data, Ytest_data,
                 input_shape=(28, 28, 1),
                 batch_size=128,
                 epochs=15,
                 optimizer=tf.keras.optimizers.SGD(),
                 loss=tf.keras.losses.binary_crossentropy,
                 metrics=None):

        if metrics is None:
            metrics = ['accuracy']  # Set metrics here if not provided

        self._modelCNN = None
        self._historyCNN = None
        self._Xtrain_data = Xtrain_data
        self._Ytrain_data = Ytrain_data
        self._Xtest_data = Xtest_data
        self._Ytest_data = Ytest_data
        self._batch_size = batch_size
        self._epochs = epochs
        self._input_shape = input_shape
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics
        return

    def CreateCNN(self):

        from keras.initializers import VarianceScaling  # Layer weight initializers

        #layers.InputLayer(my_input_shape),
        #layers.Rescaling(scale=2.0 / 255.0, offset=-1.0),

        self._modelCNN = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=self._input_shape,
                                   kernel_initializer=VarianceScaling()),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=self._input_shape,
                                   kernel_initializer=VarianceScaling()),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self._input_shape,
                                   kernel_initializer=VarianceScaling()),
            tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=self._input_shape,
                                   kernel_initializer=VarianceScaling()),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=self._input_shape,
                                   kernel_initializer=VarianceScaling()),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        self._modelCNN.summary()

        self._modelCNN.compile(self._optimizer, self._loss,
                               metrics=self._metrics)  # Pass metrics=['accuracy'] separately
        self._historyCNN = self._modelCNN.fit(self._Xtrain_data, self._Ytrain_data, self._batch_size, self._epochs,
                                              validation_data=(self._Xtest_data, self._Ytest_data))

        predictions = self._modelCNN.predict(self._Xtest_data)
        self.show_prediction(10, predictions)

    def prediction(self, Xtest_data):
        return self._modelCNN.predict(Xtest_data)

    def evaluation(self, test=None):
        # Evaluate the model
        evaluation_result = self._modelCNN.evaluate(self._Xtest_data, self._Ytest_data)
        # Print evaluation results
        print("Test Loss:", evaluation_result[0])
        print("Test Accuracy:", evaluation_result[1])
        # If test data is provided, evaluate the model on it as well
        if test is not None:
            evaluation_result = self._modelCNN.evaluate(test)  #, self._Ytest_data)
            # Print evaluation results
            print("Test Loss (from provided data):", evaluation_result[0])
            print("Test Accuracy (from provided data):", evaluation_result[1])

    def ApplyTestData(self, Xtest_data):
        # Asigură-te că modelul a fost creat anterior
        if self._modelCNN is None:
            print("Eroare: Modelul CNN nu a fost creat încă.")
            return None

        # Aplică datele de test
        predictions = self._modelCNN.predict(Xtest_data)

        return predictions

    def show_image(self, img, label, pred_label):
        plt.imshow(img, cmap='gray')
        plt.title(f'Etichetă reală: {label}, Etichetă prezisă: {pred_label}')
        plt.axis('off')

    # Afișare rezultate
    def show_predictionl(self, num_images_to_display, predictions):

        plt.figure(figsize=(10, 10))
        for i in range(num_images_to_display):
            pred_label = np.argmax(predictions[i])
            plt.imshow(self._Xtest_data[i], cmap='gray')
            plt.title(f'Etichetă reală: {self._Ytest_data[i]}, Etichetă prezisă: {pred_label}')
            plt.axis('off')
            plt.show()

    def show_prediction(self, numeric_images_to_display, predictions):
        num_images_to_display = numeric_images_to_display
        num_cols = 5
        num_rows = (num_images_to_display + num_cols - 1) // num_cols

        plt.figure(figsize=(15, 3 * num_rows))
        for i in range(num_images_to_display):
            plt.subplot(num_rows, num_cols, i + 1)
            pred_label = np.argmax(predictions[i])
            plt.imshow(self._Xtest_data[i], cmap='gray')
            # Verificați dacă datele sunt codificate one-hot

            if (isinstance(self._Ytest_data[i], np.uint8) or isinstance(self._Ytest_data[i], np.int64)):
                labels = self._Ytest_data[i]
            else:
                if len(self._Ytest_data[i]) > 1:
                    # Transformați datele înapoi în etichete întregi
                    labels = np.argmax(self._Ytest_data[i])
                else:
                    # Dacă datele nu sunt codificate one-hot, păstrați-le nemodificate
                    labels = self._Ytest_data[i]

            plt.title(f'Real: {labels}, Predicted: {pred_label}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Afisare grafic evaluare
    def model_evaluation(self):

        # Create subplots for loss and accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=100)

        # Plot training and validation loss
        ax1.plot(self._historyCNN.history['loss'], label='Training Loss')
        ax1.plot(self._historyCNN.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()

        # Plot training and validation accuracy
        ax2.plot(self._historyCNN.history['accuracy'], label='Training Accuracy')
        ax2.plot(self._historyCNN.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.show()
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        # Save the predictions probabilities
        y_test_pred = self._modelCNN.predict(self._Xtest_data)

        # Convert one-hot encoded labels back to categorical labels if needed
        # Create a Confusion Matrix
        Ytest_data = np.reshape(self._Ytest_data, (self._Ytest_data.shape[0], 1))

        Ytest_data = np.array(self._Ytest_data)
        y_pred = np.argmax(y_test_pred, axis=1)

        # Now calculate classification report
        print(classification_report_imbalanced(Ytest_data, y_pred))

        # Create a Confusion Matrix

        # calculate the confusion matrix again
        cm = confusion_matrix(Ytest_data, np.argmax(y_test_pred, axis=1))
        plt.figure(figsize=(10, 7), dpi=100)
        ax = sn.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        ax.set(xlabel="PREDICTIONS", ylabel="LABELS")
        plt.show()

    # Afisare grafic evaluare
    def model_evaluationSGD(self):

        # Create subplots for loss and accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=100)

        # Plot training and validation loss
        ax1.plot(self._historyCNN.history['loss'], label='Training Loss')
        ax1.plot(self._historyCNN.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()

        # Plot training and validation accuracy
        ax2.plot(self._historyCNN.history['accuracy'], label='Training Accuracy')
        ax2.plot(self._historyCNN.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.show()
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        # Save the predictions probabilities
        y_test_pred = self._modelCNN.predict(self._Xtest_data)

        # Convert one-hot encoded labels back to categorical labels if needed
        # Create a Confusion Matrix
        #Ytest_data = np.reshape(self._Ytest_data, (self._Ytest_data.shape[0], 1))
        # Convert the one-hot encoded labels back to categorical labels if needed
        Ytest_data = np.argmax(self._Ytest_data, axis=1)

        # Create a confusion matrix
        cm = confusion_matrix(Ytest_data, np.argmax(y_test_pred, axis=1))
        # Convert one-hot encoded labels back to categorical labels if needed
        Ytest_data = np.argmax(self._Ytest_data, axis=1)
        y_pred = np.argmax(y_test_pred, axis=1)

        # Now calculate classification report
        print(classification_report_imbalanced(Ytest_data, y_pred))

        # Create a Confusion Matrix

        # calculate the confusion matrix again
        cm = confusion_matrix(Ytest_data, np.argmax(y_test_pred, axis=1))
        plt.figure(figsize=(10, 7), dpi=100)
        ax = sn.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        ax.set(xlabel="PREDICTIONS", ylabel="LABELS")
        plt.show()
