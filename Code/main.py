import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from dataset import CelebADataset
from model import VAE
from utils import train_vae, validate_vae
from visualize import plot_loss

if __name__ == '__main__':
    
    # Define hyperparameters
    batch_size = 128
    latent_dim = 128
    lr = 0.0001
    epochs = 50

    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Load and split the CelebA dataset
    # Next line tor directory moto change kore nish
    celeba_data_path = 'path/to/your/extracted/CelebA/folder'
    dataset = CelebADataset(celeba_data_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create the VAE model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the VAE model
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs+1):
        train_loss = train_vae(model, train_loader, optimizer, epoch, epochs, device)
        val_loss = validate_vae(model, val_loader, epoch, epochs, device)
        print(f'Epoch {epoch}: train loss={train_loss:.4f}, val loss={val_loss:.4f}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save the trained model and its weights
        # Directory mon moto change kore nish
        if epoch == epochs:
            torch.save(model.state_dict(), 'models/vae.pth')
    
    # Save the epoch losses in a file
    with open('results/epoch_losses.txt', 'w') as f:
        for epoch, train_loss, val_loss in zip(range(1, epochs+1), train_losses, val_losses):
            f.write(f'Epoch {epoch}, Train loss {train_loss:.4f}, Val loss {val_loss:.4f}\n')

    # Plot the epoch losses
    plot_loss()

    print('Training complete.')