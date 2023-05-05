import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image

def cvae_loss(x, x_recon, mu, logvar, attr, attr_recon):
    # Reconstruction loss
    mse = nn.MSELoss(reduction='sum')
    recon_loss = mse(x_recon, x)
    
    # Attribute loss
    bce = nn.BCEWithLogitsLoss(reduction='sum')
    attr_loss = bce(attr_recon, attr)
    
    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    loss = recon_loss + attr_loss + kld_loss
    return loss / x.size(0)

def train_vae(model, dataloader, optimizer, epoch, num_epochs, device):

    model.train()
    train_loss = 0.0

    for images, attrs in tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}]"):
        images = images.to(device)
        attrs = attrs.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: pass images and attrs through the CVAE model to get reconstructed images, mean, log variance, and reconstructed attributes
        x_recon, mu, logvar, attr_recon = model(images, attrs)
        
        # Calculate the combined loss for the current batch
        loss = cvae_loss(images, x_recon, mu, logvar, attrs, attr_recon)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(dataloader)
    return train_loss

def validate_vae(model, dataloader, epoch, num_epochs, device):
    
    val_loss = 0.0
    batch_idx = 0
    model.eval()

    with torch.no_grad():
        for images, attrs in tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}]"):
            images = images.to(device)
            attrs = attrs.to(device)
            x_recon, mu, logvar, attr_recon = model(images, attrs) # get reconstructed images, mean, log variance, and reconstructed attributes
            loss = cvae_loss(images, x_recon, mu, logvar, attrs, attr_recon) # calculate the combined loss for the current batch
            val_loss += loss.item()
            
            # Save the first batch of images for visualization
            if batch_idx == 0:
                n = min(images.size(0), 8) # number of images to be compared
                comparison = torch.cat([images[:n], x_recon.view(-1, 3, 64, 64)[:n]]) # concatenate images and reconstructed images
                
                if os.path.exists('./Code/results') == False:
                    os.mkdir('./Code/results')

                save_image(comparison.cpu(), './Code/results/reconstruction_' + str(epoch) + '.png', nrow=n) # save the comparison image.
                                
            batch_idx += 1

    val_loss /= len(dataloader)
    return val_loss