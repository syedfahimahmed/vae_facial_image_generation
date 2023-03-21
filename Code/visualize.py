import numpy as np
import matplotlib.pyplot as plt


def plot_loss():
    # Read epoch losses from file
    with open('results/epoch_losses.txt', 'r') as f:
        lines = f.readlines()
        train_losses = [float(line.split(',')[1].split()[2]) for line in lines]
        val_losses = [float(line.split(',')[2].split()[2]) for line in lines]

    # Plot the epoch losses
    plt.plot(np.arange(len(train_losses)), train_losses, label='training_loss')
    plt.plot(np.arange(len(val_losses)), val_losses, label='validation_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()