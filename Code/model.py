import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.act = nn.ReLU()
        
        # Define the fully connected layers for mean (mu) and log variance (logvar)
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):

        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Calculate mean (mu) and log variance (logvar) using the fully connected layers
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 4*4*256)
        self.bn2 = nn.BatchNorm1d(4*4*256)
        
        self.conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)
        
        self.act = nn.ReLU()

    def forward(self, z):
        z = self.act(self.bn1(self.fc1(z)))
        z = self.act(self.bn2(self.fc2(z)))
        z = z.view(-1, 256, 4, 4)
        z = self.act(self.bn3(self.conv1(z)))
        z = self.act(self.bn4(self.conv2(z)))
        z = self.act(self.bn5(self.conv3(z)))
        z = torch.tanh(self.conv4(z))
        return z
    
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        # Calculate the standard deviation from the log variance
        std = torch.exp(0.5 * logvar)
        
        # Generate random noise with the same shape as the standard deviation
        eps = torch.randn_like(std)
        
        # Reparameterize the latent space using the calculated mean, standard deviation, and noise
        return mu + eps * std

    def forward(self, x):
        # Pass the input through the Encoder network to obtain the mean (mu) and log variance (logvar)
        mu, logvar = self.encoder(x)
        
        # Reparameterize the latent space using the mean and log variance
        z = self.reparameterize(mu, logvar)
        
        # Pass the reparameterized latent space vector through the Decoder network to reconstruct the input
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar