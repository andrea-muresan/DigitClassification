from analiza_mnist import *
from analiza_csv import *
from create_csv import *

import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # incarcam date test scrise de mana
    Analize_HAND = Hand_CSV("imagini", "CSV", 28, 28, 1)
    x_test = Analize_HAND.Normalize_CSV()
    # incarcam date analiza MNIST
    Analize_MNIST = Date_MNIST()
    Analize_MNIST.Hist_MNIST()
    Analize_MNIST.Img_MNIST()

    # incarcam date analiza CSV
    Analize_CSV = Date_CSV("train.csv", "test.csv", 28, 28, 1)
    x_test_csv_norm = Analize_CSV.Normalize_CSV_Test()

    # Normalizare MNIST
    Analize_MNIST.Normalize_MNIST()
    # antrenare MNIST epoci =15 perfomanta
    ModelSGD = Analize_MNIST.Train_MNIST_SGD(_epochs=2)
    ModelSGD.model_evaluationSGD()
    Analize_MNIST.Test_MNIST(x_test_csv_norm)
    Analize_MNIST.Test_MNIST(x_test)

    #imbunatatire performante cu ADAM acuratete 0.9967
    #antrenare MNIST epoci =15 perfomanta
    Model = Analize_MNIST.Train_MNIST_Adam(_epochs=2)
    #evaluare MNIST
    Model.model_evaluation()
    Analize_MNIST.Test_MNIST(x_test_csv_norm)
    Analize_MNIST.Test_MNIST(x_test)

    # antrenam model pe date analiza CSV epoci =15 perfomanta batch 64
    ModelC, x_test_normalized = Analize_CSV.Train_CSV(_epochs=2)

    # evaluam model CSV
    ModelC.model_evaluation()
    # testam model CSV
    Analize_CSV.Test_CSV(x_test_normalized)
    # testam pe manual model CSV
    Analize_CSV.Test_CSV_Hand(x_test)

