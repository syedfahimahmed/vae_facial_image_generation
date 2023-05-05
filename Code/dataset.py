import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_list = os.listdir(root_dir)
        self.df = pd.read_csv('./data/list_attr_celeba.csv')

        # get only image_id and smiling column
        self.df = self.df[['image_id', 'Smiling']]

        # remove the .jpg extension from the image_id column
        self.df['image_id'] = self.df['image_id'].apply(lambda x: x.split('.')[0])

        # take 50% of the dataset
        self.image_list = self.image_list[:int(0.5*len(self.image_list))]

        print(f'Number of images: {len(self.image_list)}')

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_list[idx])
        img = Image.open(img_path).convert('RGB')

        # get the image_id from the image_list
        image_id = self.image_list[idx].split('.')[0]

        # get the smiling column from the dataframe
        smiling = self.df.loc[self.df['image_id'] == image_id, 'Smiling'].values[0]

        # convert to tensor
        smiling = torch.tensor(smiling, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, smiling

def get_celeba_dataloader(root_dir, batch_size, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CelebADataset(root_dir, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    return dataloader