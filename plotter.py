import numpy as np
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

from matplotlib import pyplot as plt
import sys
import os

def create_plots(data_directory, experiment_name):
    prod_weight_norm_base = f"{experiment_name}-prod_weight_norm"
    loss_base = f"{experiment_name}-trainloss"
    train_accuracy_base = f"{experiment_name}-trainaccuracy"
    test_loss_base = f"{experiment_name}-testloss"
    test_accuracy_base = f"{experiment_name}-testaccuracy"
    spectral_norm_base = f"{experiment_name}-spectralnorm"
    inv_norm_base = f"{experiment_name}-invnorm"
    generalization_base = f"{experiment_name}-generalizationterm"


    prod_weight_norm_arr = np.load(os.path.join(data_directory, f"{prod_weight_norm_base}.npy"))
    plt.plot(prod_weight_norm_arr)
    plt.title(f"{experiment_name} Prod Weight Norms")
    plt.xlabel("Updates")
    plt.ylabel("Prod Weight Norm")
    plt.savefig(f"{prod_weight_norm_base}.png")
    plt.close()

    loss_arr = np.load(os.path.join(data_directory, f"{loss_base}.npy"))
    plt.plot(loss_arr)
    plt.title(f"{experiment_name} Loss")
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.savefig(f"{loss_base}.png")
    plt.close()

    train_accuracy_arr = np.load(os.path.join(data_directory, f"{train_accuracy_base}.npy"))
    plt.plot(train_accuracy_arr)
    plt.title(f"{experiment_name} Train Accuracy")
    plt.xlabel("Updates")
    plt.ylabel("Train Accuracy")
    plt.savefig(f"{train_accuracy_base}.png")
    plt.close()

    loss_arr = np.load(os.path.join(data_directory, f"{test_loss_base}.npy"))
    plt.plot(loss_arr)
    plt.title(f"{experiment_name} Test Loss")
    plt.xlabel("Updates")
    plt.ylabel("Test Loss")
    plt.savefig(f"{test_loss_base}.png")
    plt.close()

    test_accuracy_arr = np.load(os.path.join(data_directory, f"{test_accuracy_base}.npy"))
    plt.plot(test_accuracy_arr)
    plt.title(f"{experiment_name} Test Accuracy")
    plt.xlabel("Updates")
    plt.ylabel("Test Accuracy")
    plt.savefig(f"{test_accuracy_base}.png")
    plt.close()

    spectral_norm_arr = np.load(os.path.join(data_directory, f"{spectral_norm_base}.npy"))
    plt.plot(spectral_norm_arr)
    plt.title(f"{experiment_name} Spectral Norm")
    plt.xlabel("Updates")
    plt.ylabel("Spectral Norm")
    plt.savefig(f"{spectral_norm_base}.png")
    plt.close()


    inv_norm_arr = np.load(os.path.join(data_directory, f"{inv_norm_base}.npy"))
    plt.plot(inv_norm_arr)
    plt.title(f"{experiment_name} Inv Norm")
    plt.xlabel("Updates")
    plt.ylabel("Inv Norm")
    plt.savefig(f"{inv_norm_base}.png")
    plt.close()



    generalization_arr = np.load(os.path.join(data_directory, f"{generalization_base}.npy"))
    plt.plot(generalization_arr)
    plt.title(f"{experiment_name} Generalization")
    plt.xlabel("Updates")
    plt.ylabel("Generalization")
    plt.savefig(f"{generalization_base}.png")
    plt.close()

if __name__ == "__main__":
    data_directory = sys.argv[1]
    print(data_directory)
    experiment_name = sys.argv[2]
    print(experiment_name)
    create_plots(data_directory, experiment_name)
