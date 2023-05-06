import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class CelebADataset(Dataset):
    def __init__(self, root_dir, attributes_df, transform=None):
        self.root_dir = root_dir
        self.attributes_df = attributes_df
        self.image_list = os.listdir(root_dir)

        # take 80% of the dataset
        self.image_list = self.image_list[:int(0.8*len(self.image_list))]

        print(f'Number of images: {len(self.image_list)}')

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_list[idx])
        img = Image.open(img_path).convert('RGB')
        
        # Get the attributes for the current image
        attrs = self.attributes_df.loc[self.attributes_df["image_id"] == self.image_list[idx]].values[0][1:]
        # Change attribute values from -1 to 1 range to 0 to 1 range
        attrs = (attrs + 1) / 2
        attrs = attrs.astype(np.float32)  # Convert attributes to float32
        attrs = torch.tensor(attrs)        # Create a tensor without specifying the dtype (it will infer the dtype from the data)

        if self.transform:
            img = self.transform(img)

        return img, attrs

def get_celeba_dataloader(root_dir, attributes_df, batch_size, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CelebADataset(root_dir, attributes_df, transform) # Pass attributes_df here
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    return dataloader