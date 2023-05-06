import os
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from early_stopping import EarlyStopping
from constants import TRAIN_TRANSFORM, batch_size, latent_dim, lr, epochs

from dataset import CelebADataset
from model import CVAE
from utils import train_vae, validate_vae
from visualize import plot_loss
import pandas as pd

if __name__ == '__main__':

    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Load and split the CelebA dataset
    early_stopping = EarlyStopping(patience=10, verbose=True)
    celeba_data_path = './Code/data/img_align_celeba/img_align_celeba/'
    
    # Read the attributes CSV file
    attributes_df = pd.read_csv("./Code/data/list_attr_celeba.csv")
    attributes_df.replace(to_replace=-1, value=0, inplace=True) # Change -1 values in the dataframe to 0
    
    # Number of attributes
    n_attributes = attributes_df.shape[1] - 1
    
    dataset = CelebADataset(celeba_data_path, attributes_df, transform=TRAIN_TRANSFORM)

    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # Create the VAE model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CVAE(latent_dim, n_attributes).to(device)
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Train the VAE model
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs+1):
        train_loss = train_vae(model, train_loader, optimizer, epoch, epochs, device)
        val_loss = validate_vae(model, val_loader, epoch, epochs, device)
        print(f'Epoch {epoch}: train loss={train_loss:.4f}, val loss={val_loss:.4f}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # Save the epoch losses in a file
    with open('./Code/results/epoch_losses.txt', 'w') as f:
        for epoch, train_loss, val_loss in zip(range(1, epochs+1), train_losses, val_losses):
            f.write(f'Epoch {epoch}, Train loss {train_loss:.4f}, Val loss {val_loss:.4f}\n')

    # Plot the epoch losses
    plot_loss()

    print('Training complete.')