import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image

def vae_loss(x, x_recon, mu, logvar):
    # Reconstruction loss
    mse = nn.MSELoss(reduction='sum')
    recon_loss = mse(x_recon, x)
    
    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    loss = recon_loss + kld_loss
    return loss / x.size(0)

def train_vae(model, dataloader, optimizer, epoch, num_epochs, device):

    model.train()
    train_loss = 0.0

    for images in tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}]"):
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: pass images through the VAE model to get reconstructed images, mean, and log variance
        x_recon, mu, logvar = model(images)
        
        # Calculate the combined loss for the current batch
        loss = vae_loss(images, x_recon, mu, logvar)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(dataloader)
    return train_loss

def validate_vae(model, dataloader, epoch, num_epochs, device):
    
    val_loss = 0.0
    batch_idx = 0
    with torch.no_grad():
        for images in tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}]"):
            images = images.to(device)
            x_recon, mu, logvar = model(images)
            loss = vae_loss(images, x_recon, mu, logvar)
            val_loss += loss.item()
            
            # Save the first batch of images for visualization
            if batch_idx == 0:
                n = min(images.size(0), 8)
                comparison = torch.cat([images[:n], x_recon.view(-1, 3, 64, 64)[:n]])
                
                # Results akta folder banais
                # Directory mon moto change kore nish
                save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)
                
            batch_idx += 1

    val_loss /= len(dataloader)
    return val_loss