import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from model import VAE

def manipulate_attribute(vae, img, attr_idx, attr_value, device):
    vae.eval()
    with torch.no_grad():
        img = img.to(device).unsqueeze(0)
        mu, logvar = vae.encoder(img)
        z = vae.reparameterize(mu, logvar)
        
        # Manipulate the attribute
        z[:, attr_idx] = attr_value
        
        # Decode the manipulated latent vector
        img_recon = vae.decoder(z)
        
        return img_recon.squeeze(0).cpu()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 128
    vae = VAE(latent_dim)
    vae.load_state_dict(torch.load('./Code/saved_models/vae.pth'))
    vae.to(device)

    # Read the attributes CSV file
    attributes_df = pd.read_csv("./Code/data/img_align_celeba/list_attr_celeba.csv")
    
    # Choose an image and the attribute to manipulate
    image_id = '000001.jpg'
    attribute_name = 'Eyeglasses'
    attr_value = 1

    # Load the image and corresponding attributes
    img_path = f"./Code/data/img_align_celeba/img_align_celeba/img_align_celeba/{image_id}"
    img = Image.open(img_path).resize((64, 64)) # Resize the image to 64x64
    img_tensor = ToTensor()(img)

    # Find the index of the attribute in the dataframe
    attr_idx = attributes_df.columns.get_loc(attribute_name) - 1
    attr_value = attr_value * 2 - 1  # Convert to {-1, 1} range
    print(f"Attribute index: {attr_idx}, Attribute value: {attr_value}")

    # Manipulate the attribute
    img_recon = manipulate_attribute(vae, img_tensor, attr_idx, attr_value, device)
    
    # Visualize the original and manipulated images
    to_pil = ToPILImage()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis("off")
    
    ax2.imshow(to_pil(img_recon))
    ax2.set_title("Manipulated Image")
    ax2.axis("off")
    
    plt.show()