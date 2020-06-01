import numpy as np
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from plotter import create_plots
import os
from models import VGGnet, SimpleNet, MLP

from new_optimizer import NewOptimizer


# vary batch size keeping number of iterations consistent,
# 1. track weight norms
# 2. track variance of the gradient
# 3. frobenius distance to initialization

def load_model(num_classes_to_predict):
    # vgg = SimpleNet(num_classes_to_predict)
    low_n_units = [32*32*3, 512, num_classes_to_predict]
    high_n_units = [32*32*3, 2048, 512, 256, num_classes_to_predict]
    lowest_n_units = [32*32*3, 128, num_classes_to_predict]

    low_n_units_mnist = [28*28, 256, num_classes_to_predict]
    high_n_units_mnist = [28*28, 512, 256, 128, 64, num_classes_to_predict]

    model = MLP(high_n_units)
    return model

def get_norm_of_tensor(m, np_arr=False):
    if np_arr:
        return np.linalg.norm(m.eval(), ord='fro')
    else:
        return np.linalg.norm(m, ord='fro')

def validate(testloader, model, criterion):
    test_iter = 0
    test_loss = 0

    total_num_correct = 0
    for j, test_data in enumerate(testloader, 0):
        inputs, labels = test_data

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        predictions = outputs.max(1)[1]
        total_num_correct += predictions.eq(labels).sum().item()

        test_loss += loss.item()
        test_iter += 1

    # return loss, accuracy
    accuracy_for_epoch = (1.0*total_num_correct)/(len(testloader.dataset))
    return (1.0 * test_loss)/test_iter, accuracy_for_epoch

def run_trial_with_set_parameters(batch_size=128, num_iterations=200, model=None, lr=0.01, dataset_name="CIFAR10"):
    experiment_name = "_batchsize_" + str(batch_size)
    experiment_name += "_numiterations_" + str(num_iterations)
    experiment_name += "_dataset_" + str(dataset_name)


    if dataset_name == "MNIST":
        normalize = transforms.Normalize(mean=[0.131], std=[0.289])
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transform)
    else:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)


    print(len(trainset))
    curr_num_updates = (1.0*len(trainset))/batch_size
    num_epochs = np.ceil(num_iterations/curr_num_updates)

    print("Experiment with batch size " + str(batch_size) + " and num iterations " + str(num_iterations) + " doing one " + str(num_epochs))

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    optimizer = NewOptimizer(model.parameters(), lr=lr, p_bound=2.0)

    # (num of iterations, 1)
    loss_lst = []


    # prod weight norm list
    prod_weight_norm_lst = []

    # spectral norm list
    spectral_norm_lst = []

    # condition number list
    inv_norm_lst = []

    # test loss array
    test_losses = []

    # test accuracy
    test_accuracies = []
    train_accuracies = []

    # generalization term
    generalization_term_lst = []

    update_iter = 0

    early_stop = False

    try:

        for epoch in range(int(num_epochs)):
            for i, data in enumerate(trainloader, 0):
                if update_iter % 5 == 0:
                    print("ON ITER  " + str(update_iter))

                inputs, labels = data

                optimizer.zero_grad()

                outputs = model(inputs)

                predictions = outputs.max(1)[1]
                total_num_correct = predictions.eq(labels).sum().item()
                accuracy_for_update = (1.0 * total_num_correct) / (len(labels))
                train_accuracies.append(accuracy_for_update)

                loss = criterion(outputs, labels)

                loss.backward()

                # update the iteration count (this is how many times the loss.backward() has been called)
                update_iter += 1
                optimizer.step()

                # ------ TRAIN LOSS TRACKING -------
                loss_lst.append(loss.item())
                # ---------------------------

                # ----- TRACK TEST LOSS AND ACCURACY -------
                test_loss, test_accuracy = validate(testloader, model, criterion)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
                # ----------------------------

                # ----- WEIGHT TRACKING ---------
                weight_norms = np.array([np.linalg.norm(p.data.flatten().numpy()) for p in model.parameters()])
                prod_weight_norm = np.prod(weight_norms)
                prod_weight_norm_lst.append(prod_weight_norm)

                inv_weight_norm_params = np.array([1 + (1.0*(update_iter - 1))/(np.linalg.norm(p.data.flatten().numpy())) for p in model.parameters()])
                inv_weight_norm = np.prod(inv_weight_norm_params)
                inv_norm_lst.append(inv_weight_norm)

                weight_norms = np.array([np.linalg.norm(p.data.numpy(), ord=2) for p in model.parameters()])
                spectral_norm = np.prod(weight_norms)
                spectral_norm_lst.append(spectral_norm)
                # -------------------------------

                # ---- GENERALIZATION BOUND -----
                generalization_term = np.sqrt(spectral_norm + inv_weight_norm)
                generalization_term_lst.append(generalization_term)


                if update_iter > num_iterations:
                    early_stop = True
                    break

            if early_stop:
                break
    except KeyboardInterrupt as e:
        print("Detected keyboard interupt")

    # Save the lists per epoch
    print("Finished running trial for " + str(update_iter) + " updates")
    data_directory = "experiment_save/"

    if not os.path.exists(data_directory):
        os.mkdir(data_directory)


    loss_lst = np.array(loss_lst)
    accuracy_lst = np.array(train_accuracies)
    test_losses = np.array(test_losses)
    test_accuracies = np.array(test_accuracies)
    prod_weight_norm_lst = np.array(prod_weight_norm_lst)
    spectral_norm_lst = np.array(spectral_norm_lst)
    inv_norm_lst = np.array(inv_norm_lst)
    generalization_term_lst = np.array(generalization_term_lst)

    print("Saving model and files")
    torch.save(model, os.path.join(data_directory, str(experiment_name) + "-finalmodel.pt"))
    print("Done saving model")

    np.save(os.path.join(data_directory, str(experiment_name) + "-prod_weight_norm.npy"), prod_weight_norm_lst)
    np.save(os.path.join(data_directory, str(experiment_name) + "-trainloss.npy"), loss_lst)
    np.save(os.path.join(data_directory, str(experiment_name) + "-trainaccuracy.npy"), accuracy_lst)
    np.save(os.path.join(data_directory, str(experiment_name) + "-testloss.npy"), test_losses)
    np.save(os.path.join(data_directory, str(experiment_name) + "-testaccuracy.npy"), test_accuracies)
    np.save(os.path.join(data_directory, str(experiment_name) + "-spectralnorm.npy"), spectral_norm_lst)
    np.save(os.path.join(data_directory, str(experiment_name) + "-invnorm.npy"), inv_norm_lst)
    np.save(os.path.join(data_directory, str(experiment_name) + "-generalizationterm.npy"), generalization_term_lst)

    create_plots(data_directory, experiment_name)



if __name__ == "__main__":
    mlp_model = load_model(10)
    run_trial_with_set_parameters(batch_size=512, num_iterations=10000, model=mlp_model, lr=0.01, dataset_name="CIFAR10")
