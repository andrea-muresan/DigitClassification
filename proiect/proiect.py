import tensorflow as tf
import numpy as np

def analyze_digit():
    # Încarcă setul de date MNIST
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # X_train și y_train sunt datele de antrenament și etichetele corespunzătoare
    # X_test și y_test sunt datele de testare și etichetele corespunzătoare
    import matplotlib.pyplot as plt
    # check if the data is uniform distributed over classes
    # Calculează frecvența fiecărei clase în setul de date
    classes, counts = np.unique(y_train, return_counts=True)

    # Afișează histograma
    plt.bar(classes, counts)
    plt.xlabel('Clasă (Cifra)')
    plt.ylabel('Numărul de Exemple')
    plt.title('Distribuția Claselor în Setul de Date MNIST')
    plt.xticks(classes)  # Etichetele de pe axa x sunt clasele (0-9)
    plt.show()

    # Afișează primele câteva imagini și etichetele lor corespunzătoare

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_train[i], cmap='gray')
        plt.title(f'Label: {y_train[i]}')
        plt.axis('off')
    plt.show()

    # normalise
    X_train , X_test = X_train / 255, X_test / 255

    # transform labels into one-hot encoding
    y_trainhot = tf.keras.utils.to_categorical(y_train, 10)
    y_testhot = tf.keras.utils.to_categorical(y_test, 10)

    # input image dimensions
    # loaded mnist data are 28x28 images, but CNN requires new reshapes (28x28x1)
    img_rows, img_cols = 28, 28
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
